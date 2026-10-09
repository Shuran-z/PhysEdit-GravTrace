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
        done = {json.loads(line)["id"] for line in open(args.out)}
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
                if n:
                    state = predictor.init_state(video_path=tmp, offload_video_to_cpu=True)
                    names = list(job["objects"])
                    for k, name in enumerate(names):
                        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=k + 1, box=np.asarray(job["objects"][name], dtype=np.float32))
                    for f, ids, logits in predictor.propagate_in_video(state):
                        masks = (logits[:, 0] > 0).cpu().numpy()
                        for oid, mask in zip(ids, masks):
                            ys, xs = np.nonzero(mask)
                            if xs.size:
                                t = tracks[names[oid - 1]]
                                t["frames"].append(first + int(f))
                                t["xyxy"].append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
                    predictor.reset_state(state)
            out.write(json.dumps({"id": job["id"], "fps": fps, "image_size": list(size), "tracks": tracks}) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
