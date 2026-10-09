"""Diagnose selected worst cases with simulator-projected boxes at identical times.

Selection uses existing errors deliberately: this is root-cause analysis, not
an independent performance estimate. Neither fitter receives target gravity.
python scripts/phyediting_tail_diagnosis.py --limit 10 --workers 6
"""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phyediting_gen import _read_tracks, _sample
from gravtrace.fit import safe_fit
from gravtrace.camera import Camera, quat_to_matrix


def evaluate(job):
    metadata, sample = job
    return {**metadata, "oracle_predictions": {
        mode: safe_fit({**sample, "offset_mode": mode}) for mode in ("anchor", "centred")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    items = {r["id"]: r for r in map(json.loads, open("runs/benchmark_gravity_v1/items.jsonl"))}
    jobs = []
    for fps in (30, 15):
        old = {p["id"]: p for p in map(json.loads, open(f"runs/genfloor_pred_{fps}_agnostic.jsonl"))}
        trials = list(map(json.loads, open(f"runs/offset_full{fps}_v1.jsonl")))
        def error(r):
            return max(abs(p["gravity"] / r["gravity_target"] - 1)
                       for p in (old.get(r["id"], {}), r["predictions"]["centred"]) if p.get("status") == "ok")
        valid = [r for r in trials if old.get(r["id"], {}).get("status") == "ok"]
        selected = sorted(valid, key=error, reverse=True)[:a.limit]
        rows = {r["item_id"]: r for r in map(json.loads, open(f"runs/genfloor_rows_{fps}.jsonl"))}
        tracks = _read_tracks(f"runs/genfloor_tracks_{fps}.jsonl")
        for rec in selected:
            it, row = items[rec["id"]], rows[rec["id"]]
            sample = _sample(row, it, tracks.get(row["sample_id"], {}), "agnostic", {})
            frames = [int(round(it["window"]["start_frame"] + t * it["fps"])) for t in sample["boxes"]["times"]]
            with np.load(Path("data/compact") / (it["trajectory"] + ".npz")) as z:
                j = [str(n) for n in z["names"]].index(it["object_name"])
                corners = np.asarray(it["object"]["corners"])
                points = np.stack([z["position"][f, j] + corners @ quat_to_matrix(z["quaternion_xyzw"][f, j]).T
                                   for f in frames])
            uv = Camera.from_dict(it["camera"], it["image_size"]).project(points)
            full = np.hstack([uv.min(axis=1), uv.max(axis=1)])
            width, height = it["image_size"]
            boxes = np.clip(full, 0, [width, height, width, height])
            sample["boxes"]["xyxy"] = boxes.tolist()
            # Keep the same timestamps even when the real object is outside the image.
            # Its zero-area clipped box then cannot masquerade as an observable track.
            offscreen = int(((boxes[:, 2:] - boxes[:, :2]).min(axis=1) < 6).sum())
            jobs.append(({"id": rec["id"], "fps": fps, "gravity_target": rec["gravity_target"],
                          "reference_frames": frames, "observed_predictions": {
                              "anchor": old[rec["id"]], "centred": rec["predictions"]["centred"]},
                          "oracle_tiny_or_offscreen_frames": offscreen}, sample))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(evaluate, jobs))
    out = {"scope": "worst-case diagnostic only, selected by existing error; simulator-projected boxes replace SAM2",
           "records": results}
    Path("docs/phyediting/tail_diagnosis.json").write_text(json.dumps(out, indent=2) + "\n")
    for r in results:
        target = r["gravity_target"]
        observed = r["observed_predictions"]["centred"]
        pred = r["oracle_predictions"]["centred"]
        print(r["fps"], r["id"], "observed_pct", round(abs(observed["gravity"] / target - 1)*100, 2),
              "oracle_pct", round(abs(pred.get("gravity", 0) / target - 1)*100, 2), "status", pred["status"])


if __name__ == "__main__":
    main()
