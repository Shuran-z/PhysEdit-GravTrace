"""Compare completed full offset trials to the saved agnostic anchor controls."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def stats(records, predictions):
    errors, groups, by_event, by_gravity = [], defaultdict(list), defaultdict(list), defaultdict(list)
    failed, tails = 0, []
    for rec in records:
        p = predictions.get(rec["id"], {})
        if p.get("status") != "ok":
            failed += 1
            continue
        e = abs(p["gravity"] / rec["gravity_target"] - 1) * 100
        errors.append(e)
        groups[rec["physics_group"]].append(e)
        by_event[rec["event"]].append(e)
        by_gravity[str(rec["gravity_target"])].append(e)
        tails.append({"id": rec["id"], "error_pct": e, "frames": p["frames"]})
    def reduce(values):
        return {"n": len(values), "mean_pct": float(np.mean(values)), "max_pct": float(np.max(values))}
    return {**reduce(errors), "failed_or_bound": failed, "attempted": len(records),
            "p95_pct": float(np.percentile(errors, 95)),
            "within10_all": sum(e <= 10 for e in errors), "within20_all": sum(e <= 20 for e in errors),
            "physics_group_mean_pct": float(np.mean([np.mean(v) for v in groups.values()])),
            "successful_physics_groups": len(groups),
            "by_event": {k: reduce(v) for k, v in by_event.items()},
            "by_gravity": {k: reduce(v) for k, v in by_gravity.items()},
            "worst10": sorted(tails, key=lambda r: -r["error_pct"])[:10]}


def main():
    out = {"scope": "full tuned development set; not held out; no frame or item exclusion by error", "fps": {}}
    for fps in (30, 15):
        records = list(map(json.loads, open(f"runs/offset_full{fps}_v1.jsonl")))
        assert len(records) == len({r["id"] for r in records}) == 1038
        anchor = {p["id"]: p for p in map(json.loads, open(f"runs/genfloor_pred_{fps}_agnostic.jsonl"))}
        centred = {r["id"]: r["predictions"]["centred"] for r in records}
        out["fps"][str(fps)] = {"anchor": stats(records, anchor), "centred": stats(records, centred)}
    Path("docs/phyediting/offset_comparison.json").write_text(json.dumps(out, indent=2) + "\n")
    for fps, methods in out["fps"].items():
        for mode, m in methods.items():
            print(fps, mode, m["n"], m["mean_pct"], m["max_pct"], m["failed_or_bound"])


if __name__ == "__main__":
    main()
