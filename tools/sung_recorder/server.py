#!/usr/bin/env python3
"""Recorder server for the sung-lyric demonstrations (the dual-decode eval).

Serves a one-page studio recorder (index.html) and writes each take straight into
fixtures/human/sung/<name>/ with the manifest mambo_lab.eval.dual_decode_eval
expects — so you never hand-drop files or hand-write a manifest. Raw audio stays
git-ignored (fixtures/human/**); only derived metrics get committed. Stdlib only.

    python3 tools/sung_recorder/server.py     # then open http://localhost:8732
    make sung-recorder
"""

from __future__ import annotations

import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUNG = REPO / "fixtures" / "human" / "sung"
PORT = 8732

_SAFE = re.compile(r"[^a-z0-9_-]")


def _slug(name: str) -> str:
    s = _SAFE.sub("", (name or "").strip().lower().replace(" ", "_"))
    return s or "anon"


def _dest(name: str) -> Path:
    return SUNG / _slug(name)


def _read_manifest(dest: Path) -> dict:
    out: dict = {}
    man = dest / "manifest.jsonl"
    if man.exists():
        for ln in man.read_text().splitlines():
            if ln.strip():
                e = json.loads(ln)
                out[e["wav"]] = e
    return out


def _write_manifest(dest: Path, entries: dict) -> None:
    lines = [json.dumps(entries[k], ensure_ascii=False) for k in sorted(entries)]
    (dest / "manifest.jsonl").write_text("\n".join(lines) + "\n")


def _saved(dest: Path) -> list:
    return sorted(p.name for p in dest.glob("*.wav")) if dest.exists() else []


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 = one request per connection. Browsers never try to reuse a socket
    # the server is closing, which is the keep-alive race that surfaces as a
    # "Failed to fetch" on POST under HTTP/1.1.
    protocol_version = "HTTP/1.0"

    def _send(self, code: int, body, ctype: str = "application/json") -> None:
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b)
        self.close_connection = True

    def do_OPTIONS(self):  # noqa: N802 — CORS preflight safety net
        self._send(200, b"")

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/status":
            q = parse_qs(urlparse(self.path).query)
            self._send(200, json.dumps({"saved": _saved(_dest(q.get("name", [""])[0]))}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/save":
            self._send(404, json.dumps({"error": "not found"}))
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        try:
            q = parse_qs(parsed.query)
            fn = q["file"][0]
            assert fn.endswith(".wav") and "/" not in fn and ".." not in fn, "bad filename"
            assert body[:4] == b"RIFF" and len(body) > 44, "not a WAV body"
            dest = _dest(q.get("name", [""])[0])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / fn).write_bytes(body)
            entries = _read_manifest(dest)
            kind = q.get("kind", ["sung"])[0]
            entry = {"wav": fn, "kind": kind}
            lyric = q.get("lyric", [""])[0].strip()
            if kind == "sung" and lyric:
                entry["lyric"] = lyric
            if "intended_notes" in q and q["intended_notes"][0]:
                entry["intended_notes"] = int(q["intended_notes"][0])
                entry["tol"] = int(q.get("tol", ["1"])[0])
            entries[fn] = entry
            _write_manifest(dest, entries)
            self._send(200, json.dumps({"ok": True, "saved": _saved(dest), "dir": str(dest.relative_to(REPO))}))
        except Exception as e:  # surface to the UI, never crash the server
            self._send(400, json.dumps({"error": str(e)}))

    def log_message(self, *a):  # keep the console quiet
        pass


def main() -> None:
    SUNG.mkdir(parents=True, exist_ok=True)
    url = f"http://localhost:{PORT}"
    print(f"\n  ♪  Mambo sung-take recorder  →  {url}")
    print(f"  writing takes into  fixtures/human/sung/<name>/  (git-ignored)")
    print("  when done:  cd lab && uv run python -m mambo_lab.eval.dual_decode_eval\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n  bye!\n")


if __name__ == "__main__":
    main()
