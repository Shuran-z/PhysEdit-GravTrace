"""Samples for generated videos: each video inherits its ground-truth sample's declared scene.

A generated video continues a ground-truth scene from a conditioning frame, so camera, object,
initial state and truth come from that ground-truth sample; only the observation changes: the
video's own target masks, its frame rate, and the time of its first frame.

VIDEOS.jsonl rows: {"id", "gt_id", "masks": {"dir", "pattern", "label"}, "fps",
                    "start_s": time of the video's first frame after the ground truth's first
                               fitted frame (0 when conditioned on that frame)}

usage: generated_manifest.py GT.jsonl VIDEOS.jsonl OUT.jsonl
"""
from __future__ import annotations

import copy
import json
import sys


def main() -> None:
    gt_path, videos_path, out_path = sys.argv[1:4]
    gt = {s["id"]: s for s in map(json.loads, open(gt_path, encoding="utf-8"))}
    missing = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for video in map(json.loads, open(videos_path, encoding="utf-8")):
            if video["gt_id"] not in gt:
                missing += 1
                continue
            s = copy.deepcopy(gt[video["gt_id"]])
            s.update(id=video["id"], gt_id=video["gt_id"], fps=video["fps"], masks=video["masks"])
            s["motion"]["t0"] = [t + float(video.get("start_s", 0.0)) for t in s["motion"]["t0"]]
            s.get("window", {}).pop("start_frame", None)  # frames now count from the video's first frame
            out.write(json.dumps(s) + "\n")
    if missing:
        print(f"{missing} videos without a ground-truth sample were skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
