"""Track declared objects through a video with SAM2 and write their per-frame boxes.

Each job prompts every object with its box in one frame (normally the conditioning frame, where the
declared initial state holds) and propagates forward:

    {"id": ..., "video": PATH, "prompt_frame": 0, "objects": {"name": [x0, y0, x1, y1], ...},
     "last_frame": 209}

Output, one line per job: {"id", "fps", "image_size", "tracks": {name: {"frames": [...], "xyxy": [...]}}}.
A frame where the object's mask is empty is left out.

    python scripts/track_sam2.py JOBS.jsonl OUT.jsonl --sam2 SAM2_REPO --ckpt sam2.1_hiera_small.pt [--shard i/n]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def logit_box(logits: np.ndarray, mode: str = "pixel") -> list | None:
    """Box of the positive logit region in pixel-edge coordinates.

    The subpixel variant interpolates the zero crossing of each axis envelope.
    Pixel centres are at i + .5; clipped image edges stay exactly 0 / size.
    It adds no observations and uses no reference geometry or gravity.
    """
    if mode not in ("pixel", "subpixel"):
        raise ValueError("box mode must be pixel or subpixel")
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or not logits.size:
        raise ValueError("logits must be a nonempty 2D array")
    logits = np.where(np.isfinite(logits), logits, -1e6)
    ys, xs = np.nonzero(logits > 0)
    if not xs.size:
        return None
    if mode == "pixel":
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    def bounds(profile):
        positive = np.flatnonzero(profile > 0)
        lo, hi = int(positive[0]), int(positive[-1])
        left = 0.0 if lo == 0 else lo - .5 + (-profile[lo - 1]) / (profile[lo] - profile[lo - 1])
        right = float(len(profile)) if hi == len(profile) - 1 else hi + .5 + profile[hi] / (profile[hi] - profile[hi + 1])
        return float(left), float(right)

    x0, x1 = bounds(logits.max(axis=0))
    y0, y1 = bounds(logits.max(axis=1))
    return [x0, y0, x1, y1]


def read_frames(path: str, first: int, last: int, folder: Path) -> tuple[int, float, tuple[int, int]]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    n, size = 0, (0, 0)
    for _ in range(first, last + 1):
        ok, img = cap.read()
        if not ok:
            break
        size = (img.shape[1], img.shape[0])
        cv2.imwrite(str(folder / f"{n:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    return n, fps, size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--sam2", required=True, help="SAM2 repository (its configs are resolved from there)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--prompt", default="box", choices=["box", "box_point"])
    ap.add_argument("--box-mode", default="pixel", choices=["pixel", "subpixel", "both"],
                    help="both writes pixel tracks and tracks_subpixel from the same logits; use a separate output file")
    ap.add_argument("--claim-dir", help="shared directory: a worker claims a job by creating <dir>/<id> first, so any number "
                                         "of workers can run the same job list without repeating work")
    args = ap.parse_args()
    sys.path.insert(0, args.sam2)
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    index, count = map(int, args.shard.split("/"))
    jobs = [json.loads(line) for i, line in enumerate(open(args.jobs)) if i % count == index]
    done = set()
    if Path(args.out).exists():
        for line in open(args.out):
            rec = json.loads(line)
            if rec.get("box_mode", "pixel") != args.box_mode:
                raise ValueError("output contains a different box mode; use a separate output file")
            done.add(rec["id"])
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    predictor = build_sam2_video_predictor(args.config, args.ckpt, device=device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()
    with open(args.out, "a") as out, torch.inference_mode(), autocast:
        for job in jobs:
            if job["id"] in done:
                continue
            if args.claim_dir:
                try:
                    os.close(os.open(os.path.join(args.claim_dir, job["id"]), os.O_CREAT | os.O_EXCL))
                except FileExistsError:
                    continue
            with tempfile.TemporaryDirectory() as tmp:
                first = int(job.get("prompt_frame", 0))
                n, fps, size = read_frames(job["video"], first, int(job.get("last_frame", 10 ** 6)), Path(tmp))
                tracks = {name: {"frames": [], "xyxy": []} for name in job["objects"]}
                subpixel_tracks = {name: {"frames": [], "xyxy": []} for name in job["objects"]}
                if n:
                    state = predictor.init_state(video_path=tmp, offload_video_to_cpu=True)
                    names = list(job["objects"])
                    for k, name in enumerate(names):
                        box = np.asarray(job["objects"][name], dtype=np.float32)
                        extra = {}
                        if args.prompt == "box_point":  # also a positive click at the box centre
                            extra = dict(points=np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], dtype=np.float32),
                                         labels=np.array([1], dtype=np.int32))
                        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=k + 1, box=box, **extra)
                    for f, ids, logits in predictor.propagate_in_video(state):
                        maps = logits[:, 0].float().cpu().numpy()
                        for oid, logit_map in zip(ids, maps):
                            box = logit_box(logit_map, "pixel" if args.box_mode == "both" else args.box_mode)
                            if box is not None:
                                t = tracks[names[oid - 1]]
                                t["frames"].append(first + int(f))
                                t["xyxy"].append(box)
                                if args.box_mode == "both":
                                    sub = subpixel_tracks[names[oid - 1]]
                                    sub["frames"].append(first + int(f))
                                    sub["xyxy"].append(logit_box(logit_map, "subpixel"))
                    predictor.reset_state(state)
            out.write(json.dumps({"id": job["id"], "fps": fps, "image_size": list(size),
                                  "box_mode": args.box_mode, "tracks": tracks,
                                  **({"tracks_subpixel": subpixel_tracks} if args.box_mode == "both" else {})}) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
