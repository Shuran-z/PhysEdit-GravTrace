"""Samples for generated videos: each video inherits its ground-truth sample's declared scene.

A generated video continues a ground-truth scene from a conditioning frame, so camera, object,
initial state and truth come from that ground-truth sample; only the observation changes: the
video's own target masks (or boxes), its frame rate, and the time of its first frame.

VIDEOS.jsonl rows: {"id", "gt_id", "fps",
                    "masks": {"dir", "pattern", "label"}  or  "boxes": {"frames", "xyxy", "times"},
                    "start_s": time of the video's first frame after the ground truth's first
                               fitted frame (negative when the clip starts earlier)}

A clip that starts earlier is fitted from its first frame at or after that instant, so it covers
the same phase of the motion as the ground-truth fit.

usage: generated_manifest.py GT.jsonl VIDEOS.jsonl OUT.jsonl
"""
from __future__ import annotations

import copy
import json
import math
import sys


def first_frame_after(video: dict, instant: float) -> tuple[int, float]:
    """Index and time of the video's first frame at or after `instant` seconds."""
    if "boxes" in video:
        frames = video["boxes"]["frames"]
        times = video["boxes"].get("times") or [f / video["fps"] for f in frames]
        i = next((i for i, t in enumerate(times) if t >= instant - 1e-6), None)
        return (frames[i], times[i]) if i is not None else (10 ** 9, instant)  # too short: fits as no track
    k = math.ceil(instant * video["fps"] - 1e-6)
    return k, k / video["fps"]


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
            s.pop("masks", None)
            s.update(id=video["id"], gt_id=video["gt_id"], fps=video["fps"])
            s.update({k: video[k] for k in ("masks", "boxes") if k in video})
            shift = float(video.get("start_s", 0.0))
            window = s.setdefault("window", {})
            window.pop("start_frame", None)  # frames now count from the video's first frame
            if shift < 0.0:
                window["start_frame"], t_first = first_frame_after(video, -shift)
                shift += t_first
            s["motion"]["t0"] = [t + shift for t in s["motion"]["t0"]]
            out.write(json.dumps(s) + "\n")
    if missing:
        print(f"{missing} videos without a ground-truth sample were skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
