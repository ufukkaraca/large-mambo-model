#!/usr/bin/env bash
# Render mambo_system.svg -> .pdf (for the LaTeX/arXiv build) and .png (for the
# README / GitHub) via headless Chrome — no extra deps. Re-run after editing the SVG.
set -euo pipefail
cd "$(dirname "$0")"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W=1190; H=430
WRAP="$(mktemp -t fig).html"
trap 'rm -f "$WRAP"' EXIT
{ echo '<!doctype html><meta charset=utf-8><style>@page{size:'"$W"'px '"$H"'px;margin:0}html,body{margin:0;padding:0}svg{display:block}</style>'; cat mambo_system.svg; } > "$WRAP"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf=mambo_system.pdf "file://$WRAP" 2>/dev/null
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --default-background-color=FFFFFFFF \
  --force-device-scale-factor=2 --window-size=${W},${H} --screenshot=mambo_system.png "file://$WRAP" 2>/dev/null
echo "rendered: $(ls -1 mambo_system.pdf mambo_system.png 2>/dev/null | tr '\n' ' ')"
