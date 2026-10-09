"""Depth-probe jobs (one per video) from tracked samples: the windows' frames and boxes plus pixel intrinsics.

    python scripts/depth_jobs.py SAMPLES.jsonl OUT.jsonl --video-root ROOT [--compact data/compact]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("out")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--compact", type=Path, default=Path("data/compact"))
    args = ap.parse_args()
    jobs: dict = defaultdict(lambda: {"windows": []})
    for line in open(args.samples):
        s = json.loads(line)
        if len(s["boxes"]["frames"]) < 3:
            continue
        camera = s["meta"]["camera"]
        video = json.loads((args.compact / f"{s['gt_id']}.json").read_text())["videos"][camera]
        width, height = s["image_size"]
        job = jobs[video]
        job["video"] = f"{args.video_root}/{video}"
        job["K"] = [[s["camera"]["fx"] * width, 0, width / 2], [0, abs(s["camera"]["fy"]) * height, height / 2], [0, 0, 1]]
        job["windows"].append({"id": s["id"], "frames": s["boxes"]["frames"], "boxes": s["boxes"]["xyxy"]})
    with open(args.out, "w") as f:
        for job in sorted(jobs.values(), key=lambda j: -sum(len(w["frames"]) for w in j["windows"])):
            f.write(json.dumps(job) + "\n")
    print(len(jobs), "videos,", sum(len(j["windows"]) for j in jobs.values()), "windows")


if __name__ == "__main__":
    main()
