"""Planner (PAPER §4.7): UIR -> mambo.action.v1 via an LLM tool-use loop.

The planner is a text-domain reasoner: it never sees audio, only the UIR (forced
— Claude has no audio input — and desirable: auditable, replayable,
model-agnostic). It receives a frozen system prompt (producer persona, tool
catalog, confirmation + confidence policy), the session context, and the UIR;
it emits tool calls that become the action plan. Tool inputs are parsed with
``json``, never string matching.

Backend abstraction (DECISIONS D12): default is OpenRouter's free router
(OpenAI-compatible, tool calling verified). An Anthropic backend (SDK +
cache_control) slots in unchanged when a key exists. The planner records which
model actually answered (the free router picks one per call) for traceability.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

from . import actions, secrets
from .actions import Action, ActionPlan

SYSTEM_PROMPT = """\
You are Mambo, a music producer co-pilot. You receive a structured percept of one
studio utterance (a `mambo.utterance.v1` document: interleaved spoken-instruction
spans and hummed-melody spans with notes, key, and tempo) plus the DAW session
context. You reason like a producer and emit an ACTION PLAN as a sequence of tool
calls. You never see audio — only the percept. You never actuate anything; your
tool calls are recorded as a replayable plan.

Rules:
- Reference a melody segment's notes by its index: `notes_ref: "seg:<i>"` (the
  i-th segment in the percept). Only melody/ambiguous segments have notes.
- ANY hummed melody in the percept (a musical sketch like "give me something
  like <hum>", OR a bare hum with no words) means the user wants those notes:
  ALWAYS emit `play_preview` THEN `insert_notes` for that melody segment — never
  preview without inserting. Use the segment's own tempo
  (`analysis.tempo_bpm.value`) UNLESS a trailing modifier changes it: "slower"
  -> about 80% of it; "faster" -> about 120%; "higher/lower" -> keep the tempo
  (the user will re-hum; do not transpose).
- A mixing instruction ("kick the drums up", "make the bass louder") -> a
  relative `change_track_volume` (e.g. +2 dB); "mute"/"solo" -> mute/solo. These
  execute immediately (no preview).
- An INSTRUMENT/timbre change on an existing part ("make it electric", "on
  strings", "warmer", "on a synth", "make it a pad") -> `set_track_instrument`
  with a `patch` like "electric_piano", "strings", "synth", "warm_pad",
  "bright_piano". (Use this for changing what is already there; use a
  `play_preview` patch only when first auditioning a fresh hum.)
- Transport words ("play", "stop", "record", "go back to the start") ->
  `transport`. "loop that" / "make it a loop" / "loop it" -> `transport` with
  action "cycle". "Add a track" -> `create_track`.
- CONFIRMATION: insert/record/tempo-change are audible-in-project; do not also
  set needs_confirmation yourself — the system derives it.
- CONFIDENCE: if a chosen melody segment's `analysis.tempo_bpm.confidence` < 0.6,
  add an `ask_user` action proposing a tempo. If the top-2 key candidates are
  within 0.1 of each other, name BOTH keys in your one-line summary.
- Output: emit the tool calls in order, and put a single concise intent-summary
  sentence in your text content (no JSON, no explanation).
"""


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict[str, Any]]  # [{id, name, arguments(dict)}]
    raw_message: dict[str, Any]        # the assistant turn, for multi-turn append
    model: str


class OpenRouterBackend:
    """OpenAI-compatible chat-completions backend (OpenRouter free router)."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    # A specific capable free model (pinned for reproducibility per DECISIONS D12)
    # rather than the `openrouter/free` auto-router, which fans out across ~11
    # wildly-varying free models per the R1 ablation.
    def __init__(self, model: str = "openai/gpt-oss-120b:free", *, temperature: float = 0.0,
                 max_retries: int = 4, timeout: int = 90, max_tokens: Optional[int] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.api_key = api_key  # explicit key (from UI settings); else the .env fallback below
        if base_url:  # any OpenAI-compatible host (OpenAI, Groq, a custom server)
            self.URL = base_url.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict:
        key = self.api_key or secrets.get("OPENROUTER_API_KEY", required=True)
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        import requests

        body = {"model": self.model, "messages": messages, "tools": tools,
                "tool_choice": "auto", "temperature": self.temperature}
        if self.max_tokens:  # cap output — small local models otherwise ramble past
            body["max_tokens"] = self.max_tokens  # the timeout instead of emitting tool calls
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.URL, headers=self._headers(), json=body, timeout=self.timeout)
                if r.status_code == 429:  # rate limited — back off
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                d = r.json()
                msg = d["choices"][0]["message"]
                tcs = []
                for tc in msg.get("tool_calls") or []:
                    fn = tc["function"]
                    try:
                        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tcs.append({"id": tc.get("id", f"call_{len(tcs)}"), "name": fn["name"], "arguments": args})
                return LLMResponse(content=(msg.get("content") or "").strip(),
                                   tool_calls=tcs, raw_message=msg, model=d.get("model", self.model))
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"{type(self).__name__} call failed after {self.max_retries} retries: {last_err}")


class OllamaBackend(OpenRouterBackend):
    """OpenAI-compatible LOCAL backend — Ollama at localhost:11434 (also fits an
    MLX/llama.cpp OpenAI server via --base-url). Free, on-device, no API key: the
    budget/offline planner. Same chat-completions + tool-calling shape as
    OpenRouter, so only the URL, headers (none), and default model differ.
    Select it with env MAMBO_PLANNER_BACKEND=ollama (MAMBO_OLLAMA_MODEL / _URL)."""

    URL = "http://localhost:11434/v1/chat/completions"

    def __init__(self, model: str = "qwen2.5:7b", *, temperature: float = 0.0,
                 max_retries: int = 2, timeout: int = 300, max_tokens: int = 512,
                 base_url: Optional[str] = None):
        super().__init__(model=model, temperature=temperature, max_retries=max_retries,
                         timeout=timeout, max_tokens=max_tokens)
        if base_url:  # e.g. an MLX server on another port
            self.URL = base_url.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}  # local server needs no auth


def backend_from_settings(active: dict) -> OpenRouterBackend:
    """Build a planner backend from `settings.get_active()` (the UI-chosen provider).
    Every provider is OpenAI-compatible, so it's one class with a base URL + key;
    `local` gets the Ollama defaults (no key, output cap)."""
    model, base_url, key = active["model"], active["base_url"], active.get("key")
    if active.get("local"):
        return OllamaBackend(model=model, base_url=base_url)
    return OpenRouterBackend(model=model, api_key=key, base_url=base_url)


def _default_backend() -> OpenRouterBackend:
    """Pick the planner backend (D12 swappable backends), in precedence order:
    1) env override `MAMBO_PLANNER_BACKEND=ollama|local` (for the gate / CI);
    2) the UI settings file (`out/studio_settings.json`) if present and ready;
    3) the shipped default (OpenRouter free)."""
    import os

    which = (os.environ.get("MAMBO_PLANNER_BACKEND") or "").strip().lower()
    if which in ("ollama", "local"):
        return OllamaBackend(model=os.environ.get("MAMBO_OLLAMA_MODEL", "qwen2.5:7b"),
                             base_url=os.environ.get("MAMBO_OLLAMA_URL") or None)
    try:
        from . import settings
        if settings.PATH.exists():
            active = settings.get_active()
            if active.get("ready"):
                return backend_from_settings(active)
    except Exception:  # never let a settings glitch break planning
        pass
    return OpenRouterBackend()


def plan(uir: dict[str, Any], *, session_context: Optional[dict] = None,
         backend: Optional[OpenRouterBackend] = None, out_dir: str = "out") -> ActionPlan:
    """UIR -> validated mambo.action.v1 (note-bearing ops get an .mid artifact)."""
    backend = backend or _default_backend()
    ctx = session_context or uir.get("session_context") or {}
    user = (
        f"Session context (JSON):\n{json.dumps(ctx)}\n\n"
        f"Utterance percept (mambo.utterance.v1):\n{json.dumps(uir)}\n\n"
        "Emit the action plan as tool calls (in order) and a one-line intent summary as text."
    )
    import jsonschema

    uid = uir.get("utterance_id", "utt")
    ref, tempo = _primary_melody(uir)
    tools = actions.planner_tools()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]

    acts: list[Action] = []
    seen: set[str] = set()
    summary = ""
    model_used = backend.model
    for _step in range(6):  # tool-use loop: one tool call per turn for many models
        resp = backend.chat(messages, tools)
        model_used = resp.model
        if resp.content:
            summary = resp.content
        if not resp.tool_calls:
            break
        messages.append(resp.raw_message)  # the assistant turn (carries tool_calls)
        for tc in resp.tool_calls:
            op, args = tc["name"], dict(tc["arguments"] or {})
            if op in actions.TOOLS:
                if op in ("play_preview", "insert_notes"):
                    # Correct a notes_ref the model mis-indexed onto a non-melody
                    # segment (e.g. seg:0 = the speech span) to the real melody.
                    if not _has_notes(uir, args.get("notes_ref")) and ref:
                        args["notes_ref"] = ref
                    args.setdefault("tempo_bpm", float(tempo)) if tempo else None
                try:
                    jsonschema.validate(args, actions.TOOLS[op]["params"])
                    sig = f"{op}:{json.dumps(args, sort_keys=True)}"
                    if sig not in seen:
                        seen.add(sig)
                        acts.append(Action(op=op, args=args,
                                           artifacts={"midi_file": f"{out_dir}/{uid}.mid"} if op == "insert_notes" else None))
                except jsonschema.ValidationError:
                    pass
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": _tool_result(op)})

    ap = ActionPlan(utterance_id=uid, intent_summary=(summary or _fallback_summary(acts))[:300], actions=acts)
    ap.validate()
    ap.model_used = model_used  # type: ignore[attr-defined]  (traceability, not serialized)
    return ap


def _tool_result(op: str) -> str:
    """Synthetic tool result that nudges the model to finish the plan (these
    tools record plan steps and return immediately; no live actuation)."""
    if op == "play_preview":
        return "Preview shown to the user (recorded). If they want these notes in the project, call insert_notes; then stop."
    return "Recorded. Continue with the next plan step, or stop if the plan is complete."


def _primary_melody(uir: dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    """The first melody/ambiguous segment's ref and tempo — for arg repair."""
    for i, s in enumerate(uir.get("segments", [])):
        if s["kind"] in ("melody", "ambiguous"):
            t = (s.get("analysis", {}).get("tempo_bpm") or {}).get("value")
            return f"seg:{i}", t
    return None, None


def _has_notes(uir: dict[str, Any], ref: Optional[str]) -> bool:
    import re

    m = re.match(r"^seg:(\d+)$", ref or "")
    if not m:
        return False
    i = int(m.group(1))
    segs = uir.get("segments", [])
    return i < len(segs) and bool(segs[i].get("notes"))


def _fallback_summary(acts: list[Action]) -> str:
    ops = ", ".join(a.op for a in acts) or "no-op"
    return f"plan: {ops}"
