"""Null checks of the generated-video scoring: does the inferred gravity come from the video or from the declared state?

Two synthetic "generators" on every benchmark item, written as rows and tracks that `phyediting_gen.py score`
reads like those of a real model:

    static  the object never moves: its condition-frame box in every frame
    earth   the video always shows Earth gravity: the item's 9.81 m/s^2 sibling trajectory, projected boxes

A protocol that measures the video reports about 0.1 (the search bound) for `static` and about 9.81 for `earth`
(where the sibling is in flight during the item's window), whatever the target; one that echoes the declared
state follows the target.

    python scripts/phyediting_gen_nulls.py ITEMS OUT_DIR [--fps 16] [--compact data/compact]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phyediting_samples import prompt_boxes

CONDITION_FRAME = 23
EARTH = "g03_earth"


def sibling(trajectory: str, gravity_tag: str) -> str:
    parts = trajectory.split("__")
    parts[2] = gravity_tag
    return "__".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--compact", type=Path, default=Path("data/compact"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {k: open(args.out / f"rows_{k}.jsonl", "w") for k in ("static", "earth")}
    tracks = {k: open(args.out / f"tracks_{k}.jsonl", "w") for k in ("static", "earth")}
    missing = 0
    for line in open(args.items):
        it = json.loads(line)
        ref_fps = float(it["fps"])
        end = it["window"]["start_frame"] + it["window"]["max_frames"] - 1
        n = int(np.ceil((end - CONDITION_FRAME) / ref_fps * args.fps)) + 2
        times = CONDITION_FRAME / ref_fps + np.arange(n) / args.fps
        name, cam = it["object_name"], it["camera_id"]
        header = json.loads((args.compact / f"{it['trajectory']}.json").read_text())
        with np.load(args.compact / f"{it['trajectory']}.npz") as z:
            box = prompt_boxes(header, z, cam, frame=CONDITION_FRAME).get(name)
        boxes = {"static": [(f, box) for f in range(n)] if box else []}
        earth = sibling(it["trajectory"], EARTH)
        if (args.compact / f"{earth}.npz").exists():
            h = json.loads((args.compact / f"{earth}.json").read_text())
            with np.load(args.compact / f"{earth}.npz") as z:
                last = z["position"].shape[0] - 1
                seq = []
                for f, t in enumerate(times):
                    b = prompt_boxes(h, z, cam, frame=min(int(round(t * ref_fps)), last)).get(name)
                    if b:
                        seq.append((f, b))
            boxes["earth"] = seq
        else:
            missing += 1
            boxes["earth"] = []
        for k, seq in boxes.items():
            sid = f"{it['id']}__null_{k}"
            rows[k].write(json.dumps({"sample_id": sid, "item_id": it["id"], "condition_frame_index": CONDITION_FRAME,
                                      "output_fps": args.fps, "gravity_truth": it["gravity"]}) + "\n")
            tracks[k].write(json.dumps({"id": sid, "fps": args.fps, "tracks": {name: {"frames": [f for f, _ in seq],
                                                                                      "xyxy": [b for _, b in seq]}}}) + "\n")
    print("items without an Earth sibling:", missing)


if __name__ == "__main__":
    main()
