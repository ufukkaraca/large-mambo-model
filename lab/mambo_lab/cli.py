"""Mambo lab CLI.

  mambo uir  --file x.wav   -> mambo.utterance.v1 for the file
  mambo demo --file x.wav   -> action plan + .mid (R1)

R0 status: the joint router (probe/router/speech/fuse) is not integrated yet, so
``uir`` currently runs the melody path over the whole file as a clearly-labeled
PARTIAL diagnostic. It never fabricates speech text from unverified spans
(the ASR-is-evidence containment rule). Full file->UIR lands when the router does.
"""

from __future__ import annotations

import argparse
import json
import sys

import soundfile as sf


def _cmd_uir(args) -> int:
    from . import fuse

    audio, sr = sf.read(args.file, dtype="float32")
    uid = args.file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    utt = fuse.file_to_uir(audio, sr, strategy=args.strategy, utterance_id=uid)
    print(utt.to_json())
    return 0


def _cmd_reaper(args) -> int:
    """Drive REAPER: audio (--file) or a typed command (--text) -> action plan
    -> rendered MIDI + plan dropped into the REAPER inbox the Lua bridge watches.
    """
    from . import fuse, ir, oracle, planner
    from .daw import reaper

    ctx = oracle.default_session_context()
    if args.file:
        audio, sr = sf.read(args.file, dtype="float32")
        uid = args.file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        utt = fuse.file_to_uir(audio, sr, strategy=args.strategy, utterance_id=uid)
        uir = utt.to_dict()
    else:  # typed command -> a text-only UIR (no perception needed)
        uid = "cmd"
        utt = ir.Utterance(uid, 16000, 0.0, source="text",
                           segments=[ir.Segment("speech", 0.0, 1.0, 1.0, text=args.text)])
        uir = utt.to_dict()
    uir["session_context"] = ctx

    plan = planner.plan(uir, out_dir=str(reaper.INBOX)) if not args.oracle else oracle.oracle_plan(uir, ctx)
    plan_d = reaper.finalize_live(plan.to_dict(), uir)  # decisive: no dead-end asks, always commit a hum
    sys.stderr.write(f"── plan (model={getattr(plan,'model_used','oracle')}) ──\n")
    print(reaper.describe(plan_d))
    if args.dry_run:
        sys.stderr.write("[dry-run] not writing to REAPER inbox.\n")
        return 0
    path = reaper.submit(plan_d, uir)
    sys.stderr.write(f"→ submitted to REAPER: {path}\n  (REAPER applies it if mambo_bridge.lua is running)\n")
    return 0


def _cmd_listen(args) -> int:
    """Live push-to-talk: record from the mic, parse, and drive REAPER. The
    bridge toward the real-time app (Stage E) without the native build."""
    import os
    import signal
    import subprocess
    import tempfile

    import numpy as np

    from . import fuse, oracle, planner
    from .daw import reaper

    ctx = oracle.default_session_context()
    sys.stderr.write(
        "\n🎤  Mambo live → REAPER. Open REAPER first (you'll hear it apply each turn).\n"
        "    Hum a melody, or speak a command: 'make it electric', 'loop that',\n"
        "    'kick it up', 'mute that', 'play', 'stop'.  Ctrl-C to quit.\n"
    )
    n = 0
    while True:
        try:
            input("\n▶  Press Enter, then hum/speak…")
        except (EOFError, KeyboardInterrupt):
            sys.stderr.write("\n👋  bye!\n")
            return 0
        wav = os.path.join(tempfile.gettempdir(), f"mambo_live_{n}.wav")
        n += 1
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "avfoundation", "-i", f":{args.device}",
             "-ac", "1", "-ar", "48000", "-t", str(args.max_seconds), wav],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            input("⏺   Recording… Press Enter to STOP.")
        except KeyboardInterrupt:
            pass
        proc.send_signal(signal.SIGINT)  # ffmpeg finalizes the file cleanly on SIGINT
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        if not os.path.exists(wav) or os.path.getsize(wav) < 2000:
            sys.stderr.write("    (no audio captured — grant Terminal microphone access in System "
                             "Settings → Privacy & Security → Microphone, then re-run)\n")
            continue
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if float(np.max(np.abs(audio))) < 0.01:
            sys.stderr.write("    (barely heard you — hum/speak louder or closer to the mic)\n")
            os.remove(wav)
            continue
        utt = fuse.file_to_uir(audio, sr, strategy="joint", utterance_id="live")
        uir = utt.to_dict()
        uir["session_context"] = ctx
        # feedback: what was actually perceived
        heard_speech = " ".join(s.get("text", "") for s in uir["segments"] if s["kind"] == "speech").strip()
        n_notes = sum(len(s.get("notes", [])) for s in uir["segments"] if s["kind"] == "melody")
        sys.stderr.write(f"    heard: {heard_speech or '(no words)'}"
                         + (f"  +  {n_notes} hummed notes" if n_notes else "") + "\n")
        try:
            plan = oracle.oracle_plan(uir, ctx) if args.oracle else planner.plan(uir, out_dir=str(reaper.INBOX))
        except Exception as e:  # network/LLM hiccup -> deterministic fallback
            sys.stderr.write(f"    (planner fell back to oracle: {str(e)[:50]})\n")
            plan = oracle.oracle_plan(uir, ctx)
        plan_d = reaper.finalize_live(plan.to_dict(), uir)  # decisive: act, never dead-end
        sys.stderr.write("    " + reaper.describe(plan_d).replace("\n", "\n    ") + "\n")
        reaper.submit(plan_d, uir)
        os.remove(wav)


def _cmd_demo(args) -> int:
    from . import actions, fuse, oracle, planner

    audio, sr = sf.read(args.file, dtype="float32")
    uid = args.file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    utt = fuse.file_to_uir(audio, sr, strategy=args.strategy, utterance_id=uid)
    uir = utt.to_dict()
    uir["session_context"] = oracle.default_session_context()

    sys.stderr.write("── UIR (mambo.utterance.v1) ──\n")
    print(json.dumps(uir, indent=2, ensure_ascii=False))
    plan = planner.plan(uir, out_dir="out")
    sys.stderr.write(f"── ACTION PLAN (mambo.action.v1)  model={getattr(plan,'model_used','?')} ──\n")
    print(plan.to_json())
    written = actions.render_plan_midi(plan.to_dict(), uir, out_dir="out")
    for ref, mf in written.items():
        sys.stderr.write(f"rendered {ref} -> {mf}\n")
        # audition through a GM synth if one is available (best-effort, optional)
        import shutil
        if shutil.which("fluidsynth"):
            import subprocess
            subprocess.run(["fluidsynth", "-i", "-q", mf], capture_output=True, timeout=15)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mambo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_uir = sub.add_parser("uir", help="file -> mambo.utterance.v1")
    p_uir.add_argument("--file", required=True)
    p_uir.add_argument("--strategy", default="joint", choices=["acoustic", "linguistic", "joint"])
    p_uir.set_defaults(func=_cmd_uir)
    p_demo = sub.add_parser("demo", help="UIR -> action plan + .mid (R1)")
    p_demo.add_argument("--file", required=True)
    p_demo.add_argument("--strategy", default="joint", choices=["acoustic", "linguistic", "joint"])
    p_demo.set_defaults(func=_cmd_demo)
    p_rea = sub.add_parser("reaper", help="hum/command -> action plan -> REAPER inbox")
    p_rea.add_argument("--file", help="audio (a hum or spoken command)")
    p_rea.add_argument("--text", help="a typed command, e.g. 'loop that'")
    p_rea.add_argument("--strategy", default="joint", choices=["acoustic", "linguistic", "joint"])
    p_rea.add_argument("--oracle", action="store_true", help="deterministic planner (no LLM)")
    p_rea.add_argument("--dry-run", action="store_true", help="print actions, don't write inbox")
    p_rea.set_defaults(func=_cmd_reaper)
    p_lis = sub.add_parser("listen", help="LIVE push-to-talk mic -> action plan -> REAPER")
    p_lis.add_argument("--device", default="1", help="avfoundation audio device index (1 = MacBook Pro Microphone)")
    p_lis.add_argument("--max-seconds", type=int, default=20)
    p_lis.add_argument("--oracle", action="store_true", help="deterministic planner (no network/LLM)")
    p_lis.set_defaults(func=_cmd_listen)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
