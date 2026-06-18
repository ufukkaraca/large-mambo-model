"""Does a REASONING layer resolve communicative role where acoustic/rules routing
cannot? — the measured core of the §3.5 reasoning-novelty claim (D23).

Nokia US6476306B2 and DAWZY segment speech+hum *acoustically* and route to fixed
decoders; neither can judge *meaning*. The one judgment that needs meaning is the
§2.4 trap: when ASR hallucinates confident words over a hum (a CONFABULATION), an
acoustic/keyword router cannot tell it from a real COMMAND — but reasoning over
coherence can. This eval measures exactly that, at the layer where the novelty
lives (semantic judgment over transcript + structure), in isolation from the
already-evaluated audio front end.

Labeled role set (all from real material):
  * command       — real studio instructions  (synthetic manifest, clean speech spans)
  * demonstration — spoken frame around a hum  ("...something like ♪..♪ but slower")
  * confabulation — Whisper's hallucinated text on a hum (harvested, real ASR output)

Two classifiers score the same set: a fair RULES baseline (cue + keyword + degeneracy
heuristics — what a non-reasoning router does) and an LLM REASONING pass. We report
per-role accuracy and, as the headline, confabulation-rejection without
over-rejecting commands. Text-level, cost-bounded (~$0.02). Needs OPENAI_API_KEY.

    cd lab && uv run python -m mambo_lab.eval.semantic_reason_eval
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path

from .. import secrets

REPO = Path(__file__).resolve().parents[3]
GOLDEN = REPO / "fixtures" / "golden"
CONFAB = GOLDEN / "confabulations_whisper.json"
MANIFEST = REPO / "fixtures" / "synthetic" / "manifest.jsonl"
MODEL = "gpt-5-mini"

_DEMO_CUE = re.compile(r"\b(something like|like this|it goes|kind of|sort of|version)\b", re.I)
_CMD_KW = re.compile(r"\b(mute|solo|record|recording|louder|softer|quieter|fader|volume|reverb|"
                     r"delay|pan|tempo|bpm|drums?|bass|kick|snare|keys?|lead|track|loop|"
                     r"duplicate|delete|undo|bounce|arm|metronome|click|higher|lower|up|down)\b", re.I)


# ----- data assembly -----------------------------------------------------------

def _truth(rel: str) -> dict | None:
    p = REPO / "fixtures" / "synthetic" / rel if not rel.startswith("fixtures") else REPO / rel
    try:
        return json.loads(p.read_text())
    except OSError:
        return None


def build_set() -> list[dict]:
    items: list[dict] = []
    rows = [json.loads(x) for x in MANIFEST.read_text().splitlines() if x.strip()]
    for r in rows:
        if r.get("snr") != "clean":
            continue
        t = _truth(r["truth"])
        if not t:
            continue
        uid = r["utterance_id"]
        spoken = " ".join(s.get("text", "") for s in t["segments"] if s.get("kind") == "speech" and s.get("text"))
        has_mel = any(s.get("kind") == "melody" for s in t["segments"])
        if not spoken:
            continue
        if "command" in uid and not has_mel:
            items.append({"text": spoken, "has_hum": False, "role": "command", "uid": uid})
        elif ("like_but" in uid or "contrast" in uid or "warmer" in uid) and has_mel:
            items.append({"text": spoken, "has_hum": True, "role": "demonstration", "uid": uid})
    # cap demonstrations so the three classes are roughly balanced (commands are
    # the scarce class at ~14); keep order deterministic.
    demos = [i for i in items if i["role"] == "demonstration"]
    others = [i for i in items if i["role"] != "demonstration"]
    items = others + demos[:16]
    # real confabulations (Whisper on hums) — the span IS a hum, text is spurious
    for c in json.loads(CONFAB.read_text()):
        if c.get("confab", "").strip():
            items.append({"text": c["confab"].strip(), "has_hum": True, "role": "confabulation", "uid": c["uid"]})
    return items


# ----- rules baseline (fair, non-reasoning) ------------------------------------

def _degenerate(text: str) -> bool:
    """Hallmarks of ASR confabulation on non-speech: token repetition / very low
    lexical diversity / trivially short filler."""
    toks = re.findall(r"[a-z']+", text.lower())
    if not toks:
        return True
    uniq = len(set(toks))
    if uniq <= 2 and len(toks) >= 3:           # "be be be be", "you you you"
        return True
    top = Counter(toks).most_common(1)[0][1]
    return top / len(toks) >= 0.5 and len(toks) >= 4  # one token is >=half


def rules_role(text: str, has_hum: bool) -> str:
    if has_hum and _DEMO_CUE.search(text):
        return "demonstration"
    if _degenerate(text):
        return "confabulation"
    if _CMD_KW.search(text):
        return "command"
    # no command structure, not an obvious demo -> most likely a confabulation
    return "confabulation"


# ----- LLM reasoning pass ------------------------------------------------------

_PROMPT = (
    "You are the reasoning layer of a studio voice assistant. The speech recognizer "
    "may hallucinate words on a HUM. Given one ASR transcript and whether a hummed "
    "melody accompanies it, classify the speaker's COMMUNICATIVE ROLE:\n"
    "- command: a coherent studio instruction (e.g. 'mute the keys', 'add reverb to the bass').\n"
    "- demonstration: language framing a hum the producer wants captured (e.g. 'give me "
    "something like <hum> but slower'); requires an accompanying hum.\n"
    "- confabulation: the words are spurious ASR output over a hum — incoherent or "
    "implausible as a studio instruction (e.g. 'the moon is rising', 'be be be be').\n"
    'Reply with ONLY a JSON object: {"role": "command|demonstration|confabulation"}.'
)


def _llm_role(text: str, has_hum: bool, key: str) -> str:
    body = {"model": MODEL, "messages": [
        {"role": "system", "content": _PROMPT},
        {"role": "user", "content": f"transcript: {text!r}\nhum present: {has_hum}"}],
        "max_completion_tokens": 2000}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=60))
    msg = r["choices"][0]["message"]["content"] or ""
    m = re.search(r'"role"\s*:\s*"(\w+)"', msg)
    role = m.group(1).lower() if m else "command"
    return role, r.get("usage", {})


# ----- scoring -----------------------------------------------------------------

def _score(items: list[dict], key: str) -> dict:
    cost = 0.0
    cm_rules: Counter = Counter()
    cm_llm: Counter = Counter()
    per = []
    for it in items:
        rr = rules_role(it["text"], it["has_hum"])
        lr, u = _llm_role(it["text"], it["has_hum"], key)
        cost += u.get("prompt_tokens", 0) / 1e6 * 0.25 + u.get("completion_tokens", 0) / 1e6 * 2.0
        cm_rules[(it["role"], rr)] += 1
        cm_llm[(it["role"], lr)] += 1
        per.append({**{k: it[k] for k in ("uid", "text", "has_hum", "role")}, "rules": rr, "llm": lr})

    def acc(cm):
        tot = sum(cm.values())
        ok = sum(v for (g, p), v in cm.items() if g == p)
        return round(ok / tot, 3) if tot else None

    def per_role(cm, role):
        tot = sum(v for (g, _), v in cm.items() if g == role)
        ok = sum(v for (g, p), v in cm.items() if g == role and p == role)
        return f"{ok}/{tot}" if tot else "0/0"

    roles = ("command", "demonstration", "confabulation")
    return {
        "n": len(items),
        "rules": {"overall": acc(cm_rules), **{r: per_role(cm_rules, r) for r in roles}},
        "llm": {"overall": acc(cm_llm), **{r: per_role(cm_llm, r) for r in roles}},
        "est_cost_usd": round(cost, 4),
        "per_item": per,
    }


def main() -> int:
    secrets.load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set")
    if not CONFAB.exists():
        raise SystemExit(f"no confabulations at {CONFAB} — run the harvest first")
    items = build_set()
    counts = Counter(i["role"] for i in items)
    print(f"role set: {dict(counts)} (n={len(items)})")
    res = _score(items, key)

    print(f"\n=== Communicative-role classification — rules vs reasoning (n={res['n']}) ===")
    print(f"{'classifier':12}{'overall':>9}{'command':>10}{'demo':>9}{'confab':>9}")
    for name in ("rules", "llm"):
        s = res[name]
        lbl = "reasoning" if name == "llm" else "rules"
        print(f"{lbl:12}{s['overall']:>9.3f}{s['command']:>10}{s['demonstration']:>9}{s['confabulation']:>9}")
    print(f"\nest cost: ${res['est_cost_usd']}")

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rd = REPO / "runs" / f"{now}-semantic-reason"
    rd.mkdir(parents=True, exist_ok=True)
    res["model"] = MODEL
    res["commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res["dirty"] = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    (rd / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {rd.relative_to(REPO)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
