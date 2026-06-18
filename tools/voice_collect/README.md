# Mambo Voice Collector — contribute a voice

`mambo_voice_collector.html` is a **single self-contained page** (no server, no
install, nothing uploaded). You record the 30 prompts in your browser and click
**Download my recordings (.zip)** — one file with all your WAVs + `manifest.jsonl`,
namespaced under your name. You then send that zip back (open an issue on the repo
and a maintainer will share a drop link).

Diversity is what helps most: a range of gender, f0 range, accent, and musician /
non-musician backgrounds. Your raw audio is never redistributed — only derived
metrics are published (see [`SECURITY.md`](../../SECURITY.md)).

## The one catch: browsers only allow the mic over https or localhost

A `file://` double-click is blocked from the microphone by Chrome/Edge/Safari
(security). Use one of these to open the page in a secure context:

- **Easiest, zero account — Netlify Drop:** go to <https://app.netlify.com/drop>,
  drag `mambo_voice_collector.html` onto it → you get a public link in seconds.
  (Recordings still never leave your browser; only the page is hosted.)
- **GitHub Pages:** host the file from a fork and open its Pages URL.
- **Locally:** `make voice-collect` serves it at
  <http://localhost:8770/mambo_voice_collector.html> (secure context → mic works).

## For maintainers: ingesting the zips

```bash
mkdir -p fixtures/human/voices && cd fixtures/human/voices
unzip ~/Downloads/mambo_voices_Alex.zip      # -> voices/Alex/{*.wav, manifest.jsonl}
```

Each speaker lands in their own folder with the same manifest schema as E-LIVE
(`wav, kind, intended_notes, transcript, speaker`), ready for the multi-speaker gate
(per-speaker note-count / containment / hum-detection) — the generalization result
that turns the small-N validation into a stronger claim.
