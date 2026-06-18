# Security & privacy policy

## Reporting a vulnerability

Please report security issues privately by email to **ufukkaraca19@gmail.com** with
"SECURITY" in the subject, rather than opening a public issue. Include steps to
reproduce and the affected version/commit. We aim to acknowledge within a few days.

## Secrets

- All API keys live **only** in a git-ignored `.env` (see `.env.example`); they are
  never committed. `.env` is in `.gitignore` and the repository history is verified
  free of credentials.
- If you fork or deploy, keep `.env` out of version control and rotate any key that
  is ever pasted into a chat, log, screenshot, or shared file.
- Every key Mambo uses is optional (the core runs offline); grant the narrowest
  scope and lowest spend cap that works.

## Voice data & privacy

Mambo is a voice system, so privacy is part of its security surface:

- **Raw human audio is never committed or redistributed.** The repository ships only
  derived metrics and synthetic fixtures. Study volunteers' recordings stay private.
- The in-browser voice collector (`tools/voice_collect/`) records locally and uploads
  nothing on its own; contributors choose when to send a `.zip`.
- If you contribute audio, you confirm it is your own voice and consent to research
  use and to publication of derived (non-audio) metrics. Contributors are referred to
  by first name only, if at all.
- Don't open issues or PRs containing other people's voice recordings or any audio
  you don't have the right to share.
