"""Paired development experiment; this inspected benchmark is NOT a held-out test.

python scripts/phyediting_offset_trial.py --fps 15 --limit 24 --workers 6 --out runs/offset15
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phyediting_gen import _read_tracks, _sample
from gravtrace.fit import safe_fit


def evaluate(job):
    sample, metadata, modes = job
    results = {}
    for mode in modes:
        pred = safe_fit({**sample, "offset_mode": mode})
        results[mode] = pred
    return {**metadata, "predictions": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fps", type=int, choices=(15, 30), required=True)
    p.add_argument("--limit", type=int, default=24, help="0 evaluates every item")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("both", "anchor", "centred"), default="both")
    p.add_argument("--all-items", action="store_true", help="retain camera/background repeats; report as development data")
    p.add_argument("--no-contact-check", action="store_true", help="use the full declared flight window; paired contact-trimming diagnostic")
    a = p.parse_args()
    items = {r["id"]: r for r in map(json.loads, open("runs/benchmark_gravity_v1/items.jsonl"))}
    tracks = _read_tracks(f"runs/genfloor_tracks_{a.fps}.jsonl")
    rows = list(map(json.loads, open(f"runs/genfloor_rows_{a.fps}.jsonl")))
    # Choose by a fixed ID hash, not observed error; keep at most one camera/background per motion.
    rows.sort(key=lambda r: hashlib.sha256(("offset-trial-v1:" + r["item_id"]).encode()).hexdigest())
    seen, jobs = set(), []
    for row in rows:
        it = items[row["item_id"]]
        group = it["physics_group"]
        if group in seen and not a.all_items:
            continue
        seen.add(group)
        sample = _sample(row, it, tracks.get(row["sample_id"], {}), "agnostic", {})
        if a.no_contact_check:
            sample["window"]["contact_check"] = False
        modes = ("anchor", "centred") if a.mode == "both" else (a.mode,)
        jobs.append((sample, {"id": it["id"], "physics_group": group,
                              "gravity_target": it["gravity"], "event": it["event"], "camera": it["camera_id"]}, modes))
        if a.limit and len(jobs) >= a.limit:
            break
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool, out.with_suffix(".jsonl").open("w") as f:
        for rec in pool.map(evaluate, jobs):
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
    summary = {"scope": "development pilot, not independent validation", "fps": a.fps,
               "contact_check": not a.no_contact_check, "selected": len(jobs), "selection": "all items" if a.all_items else "fixed SHA256 order, one item per physics_group", "methods": {}}
    for mode in modes:
        good = [r for r in records if r["predictions"][mode].get("status") == "ok"]
        errors = np.array([abs(r["predictions"][mode]["gravity"] / r["gravity_target"] - 1) for r in good])
        summary["methods"][mode] = {"ok": len(good), "failed_or_bound": len(records) - len(good),
            "mean_pct": float(errors.mean() * 100) if errors.size else None,
            "max_pct": float(errors.max() * 100) if errors.size else None,
            "within10_all": int((errors <= .1).sum()), "within20_all": int((errors <= .2).sum())}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
