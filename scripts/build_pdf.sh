#!/usr/bin/env bash
# Build a submittable PDF of PAPER.md (arXiv preprint path, RELEASE.md §C).
#
# PAPER.md keeps GitHub-friendly emoji (✅ 🎤); xelatex's text fonts don't carry
# them, so we preprocess a build copy: ✅ → ✓ (renders, same "verified" meaning),
# decorative 🎤 dropped. Everything else (→ ≤ ≥ ≈ ⇒ ⊕ ◑ ♪ ♯ ✓ ✗) renders in
# "Arial Unicode MS" — the macOS broad-coverage font (swap MAINFONT on Linux,
# e.g. "Noto Serif" + a symbol fallback).
#
#   scripts/build_pdf.sh            # -> mambo.pdf
#   MAINFONT="Noto Serif" scripts/build_pdf.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="PAPER.md"
OUT="${1:-mambo.pdf}"
MAINFONT="${MAINFONT:-Arial Unicode MS}"
MONOFONT="${MONOFONT:-Menlo}"
TMP="$(mktemp -t mambo_paper).md"
trap 'rm -f "$TMP"' EXIT

# Preprocess a build copy (PAPER.md itself unchanged):
#  - drop GitHub-only ```mermaid blocks (the static Figure 1 replaces them in print);
#  - drop <!-- PDF-STRIP-START --> … <!-- PDF-STRIP-END --> regions (repo-only
#    appendices: the project roadmap + risk register, which read as design-doc
#    artifacts in a preprint);
#  - map emoji the text fonts lack -> renderable equivalents.
perl -0777 -pe 's/<!-- PDF-STRIP-START -->.*?<!-- PDF-STRIP-END -->\n?//gs; s/The detailed control\/data flow[^\n]*\n+```mermaid.*?```\n?//gs; s/```mermaid.*?```\n?//gs' "$SRC" \
  | sed -e 's/✅/✓/g' -e 's/🎤//g' > "$TMP"

pandoc "$TMP" -o "$OUT" \
  --pdf-engine=xelatex \
  --toc --toc-depth=2 \
  -V geometry:margin=1in \
  -V mainfont="$MAINFONT" \
  -V monofont="$MONOFONT" \
  -V colorlinks=true -V linkcolor=blue -V toccolor=black -V urlcolor=blue \
  -V fontsize=10pt

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
# fail loudly if any glyph was dropped (so a missing symbol never ships silently)
miss="$(pandoc "$TMP" -o /dev/null --pdf-engine=xelatex -V mainfont="$MAINFONT" \
        -V monofont="$MONOFONT" 2>&1 | grep -c 'Missing character' || true)"
[ "$miss" -eq 0 ] && echo "glyph check: clean ✓" || echo "⚠ $miss missing-glyph warnings — inspect fonts"
