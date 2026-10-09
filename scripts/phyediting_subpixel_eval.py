"""Evaluate integer and subpixel boxes from the same SAM2 logits, no reference velocity.

This is a ten-video development pilot (five selected tails, five fixed-hash
controls), NOT a whole-benchmark score or a held-out validation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phyediting_gen import _sample
from gravtrace.fit import safe_fit


def reindex(track, source_fps, fps):
    step = int(source_fps / fps)
    assert step * fps == source_fps
    out = {}
    for name, obj in track.items():
        pairs = [(int((f - 23) / step), b) for f, b in zip(obj["frames"], obj["xyxy"])
                 if f >= 23 and (f - 23) % step == 0]
        out[name] = {"frames": [f for f, _ in pairs], "xyxy": [b for _, b in pairs]}
    return {"tracks": out, "fps": fps}


def evaluate(job):
    meta, sample = job
    return {**meta, "prediction": safe_fit(sample)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("runs/subpixel_pilot_v1"))
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    selected = json.loads((a.dir / "selection.json").read_text())
    ids = selected["item_ids"]
    items = {r["id"]: r for r in map(json.loads, open("runs/benchmark_gravity_v1/items.jsonl"))}
    tracks = {r["id"]: r for r in map(json.loads, (a.dir / "tracks_both.jsonl").open())}
    expected = {items[i]["trajectory"] + "__" + items[i]["camera_id"] for i in ids}
    assert expected.issubset(tracks), "incomplete tracking: do not score a partial pilot"
    jobs = []
    for fps in (30, 15):
        rows = {r["item_id"]: r for r in map(json.loads, open(f"runs/genfloor_rows_{fps}.jsonl"))}
        for i in ids:
            it, row = items[i], rows[i]
            rec = tracks[it["trajectory"] + "__" + it["camera_id"]]
            assert rec["box_mode"] == "both" and rec["fps"] == 30
            for mode, key in (("pixel", "tracks"), ("subpixel", "tracks_subpixel")):
                sample = _sample(row, it, reindex(rec[key], rec["fps"], fps), "agnostic", {})
                sample["offset_mode"] = "centred"
                jobs.append(({"id": i, "fps": fps, "mode": mode, "gravity_target": it["gravity"],
                              "cohort": "diagnostic_tail" if ids.index(i) < 5 else "hash_control"}, sample))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(evaluate, jobs))
    summary = {}
    for fps in (30, 15):
      for cohort in ("all", "diagnostic_tail", "hash_control"):
        for mode in ("pixel", "subpixel"):
            rs = [r for r in results if r["fps"] == fps and r["mode"] == mode
                  and (cohort == "all" or r["cohort"] == cohort)]
            e = [abs(r["prediction"]["gravity"] / r["gravity_target"] - 1) * 100
                 for r in rs if r["prediction"].get("status") == "ok"]
            summary[f"{fps}_{cohort}_{mode}"] = {"attempted": len(rs), "ok": len(e), "failed_or_bound": len(rs) - len(e),
                                        "mean_pct": float(np.mean(e)) if e else None,
                                        "max_pct": float(np.max(e)) if e else None}
    out = {"scope": selected["scope"], "protocol": "same logits, same frame indices, agnostic velocity, centred offsets",
           "summary": summary, "records": results}
    Path("docs/phyediting/subpixel_pilot.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
