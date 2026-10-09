"""Recompute the common-observation comparison, including all failures in coverage.

python scripts/phyediting_common_report.py --out docs/phyediting/common_observations.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    samples = {s["id"]: s for s in map(json.loads, open("runs/common_sam2.jsonl"))}
    files = {"ours": Path("runs/common_pred_ours.jsonl")}
    centred = Path("runs/common_pred_ours_centred.jsonl")
    if centred.exists():
        files["ours_centred_experimental"] = centred
    files.update({p.stem: p for p in Path("runs/common_baselines").glob("*.jsonl")
                  if p.stem.endswith("_anchored_v0") or p.stem in ("pixel2d_matched", "lift_oracle_v0")})
    methods = {}
    for name, path in files.items():
        preds = {s["id"]: s for s in map(json.loads, path.open())}
        if name.startswith("ours"):
            assert all(preds[i]["frames"] == len(s["boxes"]["frames"]) for i, s in samples.items())
        errors, penalised = [], []
        for i, s in samples.items():
            pred = preds.get(i, {})
            if pred.get("status") == "ok":
                err = abs(pred["gravity"] / s["truth"]["gravity"] - 1)
                errors.append(err)
                penalised.append(min(err, 1.0))
            else:
                penalised.append(1.0)
        e = np.array(errors)
        methods[name] = {"n": len(samples), "ok": len(errors), "failed_or_missing": len(samples) - len(errors),
                         "mean_pct": float(e.mean() * 100) if e.size else None,
                         "max_pct": float(e.max() * 100) if e.size else None,
                         "coverage_penalised_pct": float(np.mean(penalised) * 100)}
    out = {"scope": "known-v0 development set; identical fully visible unclipped frames; contact truncation disabled",
           "caveat": "models still use different representations and geometric/spin information; oracle depth is extra truth",
           "coverage": json.load(open("runs/common_sam2.jsonl.coverage.json")), "methods": methods}
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
