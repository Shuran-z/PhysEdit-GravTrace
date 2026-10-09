"""Fixed edge ablation on the ten-video development pilot, identical actual frames."""
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


def mask_edges(sample, mode):
    """Intersect a fixed axis mask with declared visibility; preserve actual observations."""
    allowed = {"all": [True]*4, "vertical": [False, True, False, True],
               "horizontal": [True, False, True, False]}[mode]
    visible = sample["window"].get("edges_visible", {})
    sample["window"]["edges_visible"] = {str(f): [a and b for a,b in zip(visible.get(str(f), [True]*4), allowed)]
                                           for f in sample["boxes"]["frames"]}
    sample["window"]["contact_check"] = False
    return sample


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
            for mode in ("all", "vertical", "horizontal"):
                key = "tracks"
                sample = _sample(row, it, reindex(rec[key], rec["fps"], fps), "agnostic", {})
                sample["offset_mode"] = "centred"
                mask_edges(sample, mode)
                jobs.append(({"id": i, "fps": fps, "mode": mode, "gravity_target": it["gravity"],
                              **{k: it[k] for k in ("event", "camera_id", "physics_group")},
                              "cohort": "diagnostic_tail" if ids.index(i) < 5 else "hash_control"}, sample))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(evaluate, jobs))
    summary = {}
    for fps in (30, 15):
      for cohort in ("all", "diagnostic_tail", "hash_control"):
        for mode in ("all", "vertical", "horizontal"):
            rs = [r for r in results if r["fps"] == fps and r["mode"] == mode
                  and (cohort == "all" or r["cohort"] == cohort)]
            e = [abs(r["prediction"]["gravity"] / r["gravity_target"] - 1) * 100
                 for r in rs if r["prediction"].get("status") == "ok"]
            summary[f"{fps}_{cohort}_{mode}"] = {"attempted": len(rs), "ok": len(e), "failed_or_bound": len(rs) - len(e),
                                        "mean_pct": float(np.mean(e)) if e else None,
                                        "max_pct": float(np.max(e)) if e else None}
    out = {"scope": selected["scope"], "protocol": "integer boxes, same frame indices, contact truncation disabled, agnostic velocity, centred offsets, fixed edge masks",
           "summary": summary, "records": results}
    Path("docs/phyediting/edge_ablation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
