"""Generate the R1 golden suite (PAPER R1.3): ~50 utterances with a
session context and a GOLDEN mambo.action.v1 plan (from the deterministic
oracle). The LLM planner is scored against these.

Run: cd lab && uv run python ../datagen/golden.py
Deterministic and regenerable bit-exact (oracle is rule-based).
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

from mambo_lab import actions, ir, oracle

FIX = Path(__file__).resolve().parents[1] / "fixtures"
SYN = FIX / "synthetic"
OUT = FIX / "golden"

# How many of each template to include (balanced; sums to ~50).
PLAN = {"speech_command": 12, "pure_hum": 6, "like_but": 9, "can_bass": 5,
        "hum_first": 5, "contrast": 5, "warmer": 8}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    by_template = defaultdict(list)
    for tp in sorted(glob.glob(str(SYN / "truth" / "*_clean.uir.json"))):
        tmpl = "_".join(Path(tp).name.split("_")[1:-2])
        by_template[tmpl].append(tp)

    ctx = oracle.default_session_context()
    manifest = []
    n = 0
    for tmpl, count in PLAN.items():
        for tp in by_template.get(tmpl, [])[:count]:
            uir = json.load(open(tp))
            uir["session_context"] = ctx
            ir.validate(uir)
            uid = uir["utterance_id"]
            plan = oracle.oracle_plan(uir, ctx)
            plan_dict = plan.to_dict()
            actions.validate(plan_dict)
            ir.dump(uir, str(OUT / f"{uid}.uir.json"))
            actions.dump(plan_dict, str(OUT / f"{uid}.plan.json"))
            manifest.append({"utterance_id": uid, "template": tmpl,
                             "key_op": _key_op(plan_dict), "n_actions": len(plan.actions)})
            n += 1
    (OUT / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    print(f"wrote {n} golden utterances to {OUT}")
    counts = defaultdict(int)
    for m in manifest:
        counts[m["key_op"]] += 1
    print("golden key-ops:", dict(counts))


def _key_op(plan: dict) -> str:
    for a in plan["actions"]:
        if a["op"] not in ("play_preview",):
            return a["op"]
    return plan["actions"][0]["op"] if plan["actions"] else "none"


if __name__ == "__main__":
    main()
