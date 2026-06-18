"""Multi-seed R0 robustness — is the headline 0.94 a single-seed artifact?

Regenerates the whole synthetic corpus under N independent seeds and scores each
through the SAME gate evaluators (`gate.eval_router` / `gate.eval_melody`), then
reports the headline metrics as **mean ± 95% t-CI across seeds**. This answers the
paper-review open item O4 (the canonical run is one draw; quote a distribution).

Honest about n: with a handful of seeds we use a Student-t interval (df = n−1),
not a normal one, and we print n alongside every interval.

Resumable: each seed's score is written to the run dir as it finishes; a rerun
with the same `--seeds` skips finished ones. Heavy (whisper + pyin over 225
fixtures per seed ≈ 30 min/seed) — daytime only (fan curfew 00:00–06:00 CET).

    cd lab && uv run python -m mambo_lab.eval.multiseed                 # 1234,7,13,42,99
    cd lab && uv run python -m mambo_lab.eval.multiseed --seeds 1234,7  # quick 2-seed
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import gate

REPO = gate.REPO
CORPORA = REPO / "fixtures" / "_multiseed"
SNRS = ["clean", "20db", "10db"]
STRATS = ["acoustic", "linguistic", "joint"]
BOOT_SNRS = "clean,20,10"  # bootstrap.py's --snrs (label -> clean/20db/10db)

# two-sided 95% Student-t critical value by dof (n-1); 1.96 fallback for large n.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=str(REPO), capture_output=True, text=True).stdout.strip()


def _gen_corpus(seed: int) -> Path:
    """Regenerate the corpus for `seed` (bit-exact, idempotent). Skips if a
    complete corpus (manifest + every WAV) already sits on disk."""
    out = CORPORA / f"seed_{seed}"
    man = out / "manifest.jsonl"
    if man.exists():
        rows = [json.loads(x) for x in man.read_text().splitlines() if x.strip()]
        if rows and all((out / r["wav"]).exists() for r in rows):
            return out
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(REPO / "datagen" / "bootstrap.py"),
         "--out", str(out), "--seed", str(seed), "--snrs", BOOT_SNRS],
        check=True, cwd=str(REPO / "lab"),
    )
    return out


def _m(xs: list) -> float | None:
    return float(np.mean(xs)) if xs else None


def score_seed(seed: int) -> dict:
    """Generate + score one seed through the gate evaluators. Returns the per-seed
    headline means (segment F1 per strategy/SNR, note F1, key top-2)."""
    corpus = _gen_corpus(seed)
    gate.FIXTURES = corpus  # redirect the gate evaluators at this corpus
    rows = gate.load_fixtures()
    for r in rows:  # eval_router reads r["uid"]; gate_R0 normalizes it, so we must too
        r.setdefault("uid", r.get("utterance_id"))
    rt = gate.eval_router(rows)
    mel = gate.eval_melody(rows)
    # hallucination = ASR text surviving on a melody span, per arm, pooled over SNRs
    # (the structural containment the joint router enforces and the language-only arm cannot).
    halluc = {s: {"h": sum(rt[s][snr]["halluc"] for snr in SNRS),
                  "n": sum(rt[s][snr]["n"] for snr in SNRS)} for s in STRATS}
    return {
        "seed": seed,
        "router": {s: {snr: _m(rt[s][snr]["f1"]) for snr in SNRS} for s in STRATS},
        "hallucination": {s: (halluc[s]["h"] / halluc[s]["n"] if halluc[s]["n"] else None) for s in STRATS},
        "halluc_counts": halluc,
        "note_f1": {"clean": _m(mel.get("clean", {}).get("f1", [])),
                    "10db": _m(mel.get("10db", {}).get("f1", []))},
        "key_top2": (mel["clean"]["kok"] / mel["clean"]["kn"]) if mel.get("clean", {}).get("kn") else None,
        "n_fixtures": len(rows),
    }


def agg(vals: list) -> dict | None:
    """mean ± 95% t-CI across seeds (df = n−1). None-safe."""
    xs = [v for v in vals if v is not None]
    n = len(xs)
    if n == 0:
        return None
    mean = float(np.mean(xs))
    if n == 1:
        return {"mean": mean, "std": 0.0, "lo": mean, "hi": mean, "n": 1}
    sd = float(np.std(xs, ddof=1))
    h = float(_T95.get(n - 1, 1.96) * sd / np.sqrt(n))
    return {"mean": mean, "std": sd, "lo": mean - h, "hi": mean + h, "n": n}


def _fmt(a: dict | None) -> str:
    if not a:
        return "      —"
    return f"{a['mean']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]"


def _print(agg_out: dict) -> None:
    n = agg_out["n_seeds"]
    print(f"\nMulti-seed R0 robustness — {n} seeds {agg_out['seeds']}  (mean ± 95% t-CI, df={n-1})\n")
    print(f"{'metric':28s}{'clean':>22s}{'20 dB':>22s}{'10 dB':>22s}")
    print("-" * 94)
    for st in STRATS:
        s = agg_out["strategies"][st]
        tag = f"segment F1 — {st}"
        print(f"{tag:28s}{_fmt(s['clean']):>22s}{_fmt(s['20db']):>22s}{_fmt(s['10db']):>22s}")
    print("-" * 94)
    nf = agg_out["note_f1"]
    print(f"{'note F1':28s}{_fmt(nf['clean']):>22s}{'':>22s}{_fmt(nf['10db']):>22s}")
    print(f"{'key top-2':28s}{_fmt(agg_out['key_top2']):>22s}")
    if "hallucination_pooled" in agg_out:
        print("-" * 94)
        for st in STRATS:
            p = agg_out["hallucination_pooled"][st]
            rate = f"{p['h']/p['n']*100:.1f}% ({p['h']}/{p['n']})" if p["n"] else "—"
            print(f"{'hallucination — '+st:28s}{rate:>22s}")


def _sign_p(wins: int, n: int) -> float:
    """Two-sided exact binomial (sign) test p-value, p0 = 0.5."""
    import math
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_stats(per_seed: list[dict], a: str, b: str) -> dict:
    """PAIRED comparison of arm `a` vs `b` across seeds (the correct test — the
    marginal CIs can overlap while every per-seed difference is one-signed). For
    each SNR: per-seed diff a−b, mean, win count, exact sign-test p, paired
    bootstrap 95% CI on the mean diff, and Cohen's d_z."""
    out = {}
    for snr in SNRS:
        diffs = [r["router"][a][snr] - r["router"][b][snr] for r in per_seed]
        n = len(diffs)
        md = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        wins = sum(d > 0 for d in diffs)
        rng = np.random.default_rng(1234)
        boots = [float(np.mean(rng.choice(diffs, n, replace=True))) for _ in range(5000)]
        out[snr] = {"mean_diff": round(md, 4), "wins": f"{wins}/{n}",
                    "sign_p": round(_sign_p(wins, n), 4),
                    "boot_ci": [round(float(np.percentile(boots, 2.5)), 4),
                                round(float(np.percentile(boots, 97.5)), 4)],
                    "cohen_dz": round(md / sd, 2) if sd > 0 else None}
    return out


def _print_paired(paired: dict, n_seeds: int) -> None:
    print(f"\nPaired arm comparison across {n_seeds} seeds (joint − arm; positive = joint better)\n")
    print(f"{'comparison':24s}{'SNR':>7s}{'mean Δ':>10s}{'wins':>7s}{'sign p':>9s}{'boot 95% CI':>20s}{'d_z':>7s}")
    print("-" * 84)
    for cmp, by_snr in paired.items():
        for snr, s in by_snr.items():
            ci = f"[{s['boot_ci'][0]:+.3f},{s['boot_ci'][1]:+.3f}]"
            print(f"{cmp:24s}{snr:>7s}{s['mean_diff']:>+10.3f}{s['wins']:>7s}{s['sign_p']:>9.3f}{ci:>20s}{str(s['cohen_dz']):>7s}")
    print("\nNote: at n=5 seeds the two-sided sign test floors at p=0.0625 (all-agree); "
          "≥6 seeds are needed for p<0.05. The paired bootstrap CI + d_z are the headline.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1234,7,13,42,99")
    ap.add_argument("--analyze", default=None,
                    help="post-process an existing multiseed run dir (paired tests, NO compute)")
    ap.add_argument("--out", default=None,
                    help="reuse a run dir (resume cached seeds, only compute new ones)")
    args = ap.parse_args()

    if args.analyze:
        run = REPO / args.analyze if not args.analyze.startswith("/") else __import__("pathlib").Path(args.analyze)
        agg_in = json.loads((run / "aggregate.json").read_text())
        per = agg_in["per_seed"]
        paired = {"joint_vs_linguistic": paired_stats(per, "joint", "linguistic"),
                  "joint_vs_acoustic": paired_stats(per, "joint", "acoustic")}
        (run / "paired.json").write_text(json.dumps(paired, indent=2))
        _print_paired(paired, agg_in["n_seeds"])
        print(f"\nwrote {run.relative_to(REPO)}/paired.json")
        return 0

    seeds = [int(s) for s in args.seeds.split(",")]

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = (REPO / args.out) if args.out else (REPO / "runs" / f"{ts}-multiseed")
    run.mkdir(parents=True, exist_ok=True)

    recs: list[dict] = []
    for s in seeds:
        sf = run / f"seed_{s}.json"
        if sf.exists():
            recs.append(json.loads(sf.read_text()))
            print(f"[skip] seed {s} already scored", flush=True)
            continue
        print(f"[seed {s}] regenerate + score (whisper+pyin, ~30 min) …", flush=True)
        rec = score_seed(s)
        sf.write_text(json.dumps(rec, indent=2))
        recs.append(rec)
        jc = rec["router"]["joint"]["clean"]
        print(f"[seed {s}] joint segment F1 clean={jc:.3f}  note F1 clean={rec['note_f1']['clean']:.3f}", flush=True)

    out = {
        "phase": "R0-multiseed",
        "timestamp": ts,
        "commit": _git("rev-parse", "HEAD"),
        # dirty = uncommitted edits to TRACKED code/fixtures (what affects
        # reproducibility), NOT this run's own fresh output files in runs/ —
        # so --untracked-files=no (else the seed_*.json we write mid-run, which
        # are untracked until committed, would always flag the run dirty).
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "seeds": seeds,
        "n_seeds": len(recs),
        "strategies": {st: {snr: agg([r["router"][st][snr] for r in recs]) for snr in SNRS} for st in STRATS},
        "hallucination": {st: agg([r.get("hallucination", {}).get(st) for r in recs]) for st in STRATS},
        "hallucination_pooled": {st: {
            "h": sum(r.get("halluc_counts", {}).get(st, {}).get("h", 0) for r in recs),
            "n": sum(r.get("halluc_counts", {}).get(st, {}).get("n", 0) for r in recs),
        } for st in STRATS},
        "note_f1": {k: agg([r["note_f1"][k] for r in recs]) for k in ("clean", "10db")},
        "key_top2": agg([r["key_top2"] for r in recs]),
        "per_seed": recs,
    }
    (run / "aggregate.json").write_text(json.dumps(out, indent=2))
    _print(out)
    print(f"\nwrote {run.relative_to(REPO)}/aggregate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
