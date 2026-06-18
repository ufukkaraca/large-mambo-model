"""Mambo Studio — a browser cockpit for live voice+hum commanding of REAPER.

    cd lab && uv run python -m mambo_lab.studio     # opens http://localhost:8765

Tap the mic, speak a command and/or hum a melody, tap to stop. The pipeline
(fuse → plan → finalize_live → REAPER inbox) runs and the page shows what was
heard and what REAPER was told to do — the product surface over the same core
the gates validate. The deterministic oracle planner is the default (no network);
add ?planner=1 to the URL to use the LLM planner.

This is decoupled from REAPER exactly like the CLI: it writes the action plan +
MIDI into the inbox the Lua bridge watches. Open REAPER (with the bridge) to see
each turn applied; without it, Studio still shows what *would* happen.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
PORT = 8765


# Longest-first so "jot this down" wins over "jot down".
LYRIC_CUES = ("jot this down", "jot that down", "jot it down", "jot down",
              "write this down", "write that down", "write it down", "write down",
              "note this down", "note that down", "note down",
              "remember this line", "remember the line", "add a line",
              "for the lyrics", "for the lyric", "lyrics:", "lyric:")


def _maybe_lyric(heard: str):
    """A 'jot this down …' utterance captures the rest of the line as a lyric,
    not a DAW command — the artist's notebook lives next to the console."""
    low = heard.lower()
    for cue in LYRIC_CUES:
        i = low.find(cue)
        if i != -1:
            line = heard[i + len(cue):].lstrip(" :,-—–").strip()
            line = re.sub(r"^(this|that|it|the line)\b[\s:,-]*", "", line, flags=re.I).strip()
            return line or heard
    return None


def _maybe_take(heard: str):
    """Take-log voice commands (session-level, not a DAW op): keep / scratch /
    recall / label a take. Returns an intent dict or None. Checked before planning,
    like _maybe_lyric — 'keep that' must not become a DAW action."""
    low = heard.lower()
    m = re.search(r"\btake\s+(\d+)\b", low)
    n = int(m.group(1)) if m else None
    if any(p in low for p in ("scratch that take", "scratch this take", "drop that take", "delete that take")):
        return {"op": "scratch"}
    if any(p in low for p in ("play take", "go back to", "go to take", "recall take", "play the keeper", "back to the keeper")):
        return {"op": "recall", "ref": n if n else "keeper"}
    if any(p in low for p in ("keep that", "keep this", "keep take", "that's a keeper", "thats a keeper", "that's the one")):
        return {"op": "keep", "take": n}
    if any(p in low for p in ("slate this as", "label this take", "name this take", "call this take")):
        lab = re.split(r"slate this as|label this take|name this take|call this take", low, 1)[-1].strip(" :,-.")
        return {"op": "label", "label": lab or None}
    return None


def _apply_take(intent: dict) -> dict:
    from . import session as ss
    slug = ss.current_slug()
    if not slug:
        return {"ok": True, "kind": "take", "msg": "no active session — say 'new session …' first"}
    op = intent["op"]
    if op == "keep":
        t = ss.mark_take_kept(slug)
        return {"ok": True, "kind": "take", "msg": f"kept take {t['id']}" if t else "no take to keep yet"}
    if op == "label":
        t = ss.mark_take_kept(slug, label=intent.get("label"))
        return {"ok": True, "kind": "take", "msg": f"take {t['id']} → “{t.get('label')}”" if t else "no take to label"}
    if op == "scratch":
        t = ss.scratch_take(slug)
        return {"ok": True, "kind": "take", "msg": f"scratched take {t['id']}" if t else "no take to scratch"}
    if op == "recall":
        t = ss.find_take(slug, intent["ref"])
        return {"ok": True, "kind": "take", "msg": f"recall take {t['id']}" if t else "no such take"}
    return {"ok": True, "kind": "take", "msg": "?"}


def _reaper_connected(inbox) -> bool:
    """True if the REAPER bridge wrote a fresh heartbeat (it is running + watching)."""
    try:
        return (time.time() - float((inbox / ".heartbeat").read_text().strip())) < 5.0
    except Exception:
        return False


def _capture_sung_lyric(audio, sr, uir: dict) -> tuple[dict, str | None]:
    """P3 dual-decode (D23), live. For each hummed span, re-run ASR on just that
    slice; the reasoning layer promotes it to `ambiguous` (notes + lyric) iff it
    judges the words a real SUNG lyric — else it stays melody-only and the 0%
    containment holds. Returns (uir, captured-lyric-or-None). Works offline (the
    judge falls back to rules without an API key)."""
    from . import dual_decode, probe
    pr = probe.transcribe(audio, sr)  # one whole-clip ASR; carries word timing
    span_text = {}
    for i, s in enumerate(uir.get("segments", [])):
        if s.get("kind") == "melody" and s.get("notes"):
            cand = dual_decode.candidate_from_probe(pr, s["t0"], s["t1"])
            if cand:
                span_text[i] = cand
    if not span_text:
        return uir, None
    uir = dual_decode.promote(uir, span_text)
    lyrics = [s.get("text", "") for s in uir["segments"]
              if s.get("kind") == "ambiguous" and s.get("text")]
    return uir, (" / ".join(l for l in lyrics if l) or None)


def _process(wav_bytes: bytes, use_planner: bool) -> dict:
    """Audio turn → UIR → plan → REAPER inbox; return a feedback card."""
    from . import fuse, oracle, planner  # noqa: F401  (planner lazy)
    from .daw import reaper

    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if float(np.max(np.abs(audio))) < 0.01:
        return {"ok": False, "msg": "barely heard you — a touch louder or closer to the mic"}

    from . import semantic_verify, session as session_state, settings as st, voiceprint as vp_mod
    ctx = session_state.load_session_context()  # current session, or cold-start template
    vp = vp_mod.Voiceprint.from_dict(st.get_voiceprint())  # per-user calibration (1.0 if none)
    uir = fuse.file_to_uir(audio, sr, strategy="joint", utterance_id="live",
                           pitch_step=vp.pitch_step).to_dict()
    uir = semantic_verify.verify(uir)  # P1: drop phantom melody on spoken commands (D23)
    uir, sung_lyric = _capture_sung_lyric(audio, sr, uir)  # P3: dual-decode a sung demo → notes + lyric
    uir["session_context"] = ctx
    heard = " ".join(s.get("text", "") for s in uir["segments"] if s["kind"] == "speech").strip()
    pitches = [n.get("name", "") for s in uir["segments"] if s["kind"] in ("melody", "ambiguous")
               for n in s.get("notes", [])]
    notes = len(pitches)

    lyric = _maybe_lyric(heard) if heard else None
    if lyric:  # notebook intent, not a DAW command
        return {"ok": True, "kind": "lyric", "heard": heard, "lyric": lyric,
                "notes": notes, "pitches": pitches}

    take_cmd = _maybe_take(heard) if heard else None
    if take_cmd:  # take-log intent (keep/scratch/recall) — session, not the DAW
        r = _apply_take(take_cmd)
        r["heard"] = heard
        return r

    used = "oracle"
    try:
        if use_planner:
            plan = planner.plan(uir, out_dir=str(reaper.INBOX))
            used = getattr(plan, "model_used", "planner")
        else:
            plan = oracle.oracle_plan(uir, ctx)
    except Exception as e:  # network/LLM hiccup → deterministic fallback
        plan = oracle.oracle_plan(uir, ctx)
        used = f"oracle (planner fell back: {str(e)[:40]})"

    plan_d = reaper.finalize_live(plan.to_dict(), uir)
    reaper.submit(plan_d, uir)
    # F2: log a recorded take to the active session's take history
    if any(a["op"] == "record_take" for a in plan_d.get("actions", [])):
        from . import session as session_state
        # recording IS the session starting — auto-open one if none is active,
        # so the take log (and the Takes panel) populates without a setup step.
        slug = session_state.current_slug() or session_state.create("Untitled Session")["slug"]
        ra = next(a for a in plan_d["actions"] if a["op"] == "record_take")
        session_state.append_take(slug, {k: ra["args"].get(k) for k in ("track", "label", "section")})
    did = []
    for a in plan_d.get("actions", []):
        fn = reaper._VERB.get(a["op"], lambda x: a["op"])
        try:
            did.append(fn(a.get("args", {})))
        except Exception:
            did.append(a["op"])
    if sung_lyric:  # P3: jot the captured sung lyric into the active session's Notebook
        try:
            from . import session as session_state
            slug = session_state.current_slug() or session_state.create("Untitled Session")["slug"]
            nb = session_state.read_notebook(slug)
            session_state.write_notebook(slug, (nb + ("\n" if nb else "") + "♪ " + sung_lyric).strip())
        except Exception:
            pass
    return {"ok": True, "heard": heard, "notes": notes, "pitches": pitches,
            "lyric_captured": sung_lyric,
            "intent": plan_d.get("intent_summary", ""), "did": did, "engine": used}


def _session_op(op: str, payload: dict) -> dict:
    """Backend for `POST /session?op=…`. Save/load/list/new/open/switch/rename a
    named session + its Notebook text (F1). All ops route through `session.py`
    (the `mambo.session.v1` state store); no contract schema is touched."""
    from . import session as session_state
    if op in ("save", "update"):
        slug = payload.get("slug") or session_state.current_slug()
        if not slug:  # first save with no active session → create from name
            sess = session_state.create(payload.get("name", "untitled"))
            slug = sess["slug"]
        else:
            sess = session_state.load(slug)
        for k in ("name", "phase", "project_tempo_bpm", "project_key", "transport"):
            if k in payload:
                sess[k] = payload[k]
        session_state.save(sess, notebook=payload.get("notebook"))
        return {"ok": True, "session": sess}
    if op in ("new", "create"):
        sess = session_state.create(payload.get("name", "untitled"),
                                    seed=payload.get("seed"))
        if "notebook" in payload:
            session_state.write_notebook(sess["slug"], payload["notebook"])
        return {"ok": True, "session": sess}
    if op in ("load", "open", "switch"):
        slug = payload.get("slug") or session_state.slugify(payload.get("name", ""))
        sess = session_state.switch(slug)
        return {"ok": True, "session": sess,
                "notebook": session_state.read_notebook(slug)}
    if op == "rename":
        slug = payload.get("slug") or session_state.current_slug()
        return {"ok": True, "session": session_state.rename(slug, payload["name"])}
    if op == "list":
        return {"ok": True, "sessions": session_state.list_sessions()}
    if op == "take_update":  # per-row keep ★ / relabel a specific take by id
        slug = payload.get("slug") or session_state.current_slug()
        t = session_state.set_take(slug, int(payload["id"]),
                                   kept=payload.get("kept"), label=payload.get("label"))
        return {"ok": t is not None, "take": t,
                "takes": session_state.read_takes(slug)}
    if op == "take_scratch":  # per-row ✕ a specific take by id
        slug = payload.get("slug") or session_state.current_slug()
        t = session_state.remove_take(slug, int(payload["id"]))
        return {"ok": t is not None, "dropped": t,
                "takes": session_state.read_takes(slug)}
    if op == "open_reaper":  # launch REAPER on THIS project's own .rpp document
        slug = payload.get("slug") or session_state.current_slug()
        if not slug:
            return {"ok": False, "msg": "no active project"}
        rpp = session_state.reaper_project_path(slug, create=True)
        launched, msg = False, f"project document: {rpp}"
        if sys.platform == "darwin":
            try:
                subprocess.Popen(["open", "-a", "REAPER", str(rpp)])
                launched, msg = True, f"opened {rpp.name} in REAPER"
            except Exception as e:  # REAPER not installed / open failed
                msg = f"could not launch REAPER ({e}); document is at {rpp}"
        return {"ok": True, "launched": launched, "rpp": str(rpp), "msg": msg}
    if op == "notebook":  # save notebook only
        slug = payload.get("slug") or session_state.current_slug()
        if not slug:
            return {"ok": False, "msg": "no active session"}
        session_state.write_notebook(slug, payload.get("notebook", ""))
        return {"ok": True}
    return {"ok": False, "msg": f"unknown session op: {op}"}


# ── voice calibration (the onboarding "voiceprint") ──────────────────────────
_CALIB: dict[str, list] = {}  # slot -> [(audio, sr)] accumulated across the 3 prompts


def _calib_summary() -> dict:
    from . import settings as st, voiceprint as vp_mod
    saved = st.get_voiceprint()
    vp = vp_mod.Voiceprint.from_dict(saved)
    return {"calibrated": saved is not None,
            "f0_min": round(vp.f0_min), "f0_max": round(vp.f0_max),
            "vibrato_semitones": round(vp.vibrato_semitones, 2),
            "pitch_step": round(vp.pitch_step, 2),
            "wide_vibrato": vp.pitch_step > 1.0,
            "collected": sorted(_CALIB.keys())}


def _calibrate_op(op: str, slot: str, wav: bytes) -> dict:
    """Onboarding: accumulate a clip per slot, then derive + persist the voiceprint."""
    from . import settings as st, voiceprint as vp_mod
    if op == "reset":  # back to the shipped (uncalibrated) default
        _CALIB.clear()
        d = st.load()
        d.pop("voiceprint", None)
        st.save(d)
        return {"ok": True, **_calib_summary()}
    if op == "clip" and wav:
        audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        _CALIB.setdefault(slot, []).append((audio, sr))
        return {"ok": True, **_calib_summary()}
    if op == "derive":
        held = _CALIB.get("steady", []) + _CALIB.get("range", [])
        speech = _CALIB.get("speech", [])
        if not held:
            return {"ok": False, "msg": "no hum captured — record the hum prompts first"}
        vp = vp_mod.derive(held, speech, label="you")
        st.set_voiceprint(vp.to_dict())
        _CALIB.clear()
        return {"ok": True, **_calib_summary()}
    return {"ok": False, "msg": f"unknown calibrate op: {op}"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b)
        self.close_connection = True

    def do_GET(self):  # noqa: N802
        from . import session as session_state
        from .daw import reaper
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send(200, (HERE / "studio_ui.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/status":
            ctx = session_state.load_session_context()
            self._send(200, json.dumps({
                "tracks": [t["name"] for t in ctx.get("tracks", [])],
                "inbox": str(reaper.INBOX),
                "reaper": _reaper_connected(reaper.INBOX),
                "session": session_state.current_slug(),
            }))
        elif path == "/sessions":
            self._send(200, json.dumps({"ok": True, "sessions": session_state.list_sessions(),
                                        "current": session_state.current_slug()}))
        elif path == "/session/notebook":
            slug = parse_qs(parsed.query).get("slug", [None])[0] or session_state.current_slug()
            self._send(200, json.dumps({"ok": True, "slug": slug,
                                        "notebook": session_state.read_notebook(slug) if slug else ""}))
        elif path == "/session/takes":
            slug = parse_qs(parsed.query).get("slug", [None])[0] or session_state.current_slug()
            self._send(200, json.dumps({"ok": True, "slug": slug,
                                        "takes": session_state.read_takes(slug) if slug else []}))
        elif path == "/settings":  # planner provider/model/key chooser (keys masked)
            from . import settings as st
            self._send(200, json.dumps({"ok": True, **st.public()}))
        elif path == "/calibrate":  # the voiceprint onboarding status
            self._send(200, json.dumps({"ok": True, **_calib_summary()}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        if parsed.path == "/session":
            try:
                op = parse_qs(parsed.query).get("op", [""])[0]
                payload = json.loads(body.decode() or "{}") if body else {}
                self._send(200, json.dumps(_session_op(op, payload)))
            except Exception as e:
                traceback.print_exc()
                self._send(400, json.dumps({"ok": False, "msg": f"error: {e}"}))
            return
        if parsed.path == "/settings":  # save the planner provider/model/key
            try:
                from . import settings as st
                p = json.loads(body.decode() or "{}") if body else {}
                st.set_active(p["provider"], model=p.get("model"), key=p.get("key"),
                              base_url=p.get("base_url"))
                self._send(200, json.dumps({"ok": True, **st.public()}))
            except Exception as e:
                traceback.print_exc()
                self._send(400, json.dumps({"ok": False, "msg": f"error: {e}"}))
            return
        if parsed.path == "/calibrate":  # voiceprint onboarding (clip/derive/reset)
            try:
                q = parse_qs(parsed.query)
                op = q.get("op", ["clip"])[0]
                slot = q.get("slot", [""])[0]
                self._send(200, json.dumps(_calibrate_op(op, slot, body)))
            except Exception as e:
                traceback.print_exc()
                self._send(400, json.dumps({"ok": False, "msg": f"error: {e}"}))
            return
        if parsed.path != "/command":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            use_planner = parse_qs(parsed.query).get("planner", ["0"])[0] == "1"
            self._send(200, json.dumps(_process(body, use_planner)))
        except Exception as e:
            traceback.print_exc()
            self._send(400, json.dumps({"ok": False, "msg": f"error: {e}"}))

    def log_message(self, *a):
        pass


def main(argv=None) -> int:
    # Warm the heavy models once so the first command isn't slow.
    sys.stderr.write("  🎹  Mambo Studio — warming the perception models…\n")
    try:
        from . import probe
        probe.transcribe(np.zeros(16000, dtype="float32"), 16000)
    except Exception:
        pass
    url = f"http://localhost:{PORT}"
    sys.stderr.write(f"  ready → {url}\n"
                     "  open REAPER (with the bridge) to see turns applied; tap the mic and go.\n\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n  bye!\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
