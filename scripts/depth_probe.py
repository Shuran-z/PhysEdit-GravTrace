"""Depth of a tracked object from monocular / multi-view depth models (baselines for GravTrace).

For each video job, the frames used by its windows are run through a depth model once; each window then gets
the median depth inside the central half of its tracked box, frame by frame. Jobs (one line per video):

    {"video": PATH, "K": 3x3 pixel intrinsics, "windows": [{"id", "frames": [...], "boxes": [[x0, y0, x1, y1], ...]}]}

Output, one line per (model, window): {"id", "model", "frames", "depth": [...], "relative": bool}.
Monocular models see one frame at a time; VGGT sees each window's frames together. Where a model accepts
intrinsics (MoGe-2, UniDepthV2) it is given the known ones.

    python scripts/depth_probe.py JOBS.jsonl OUT.jsonl --model depthpro --weights DIR [--code DIR ...] [--shard i/n]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

LOADERS = {}


def loader(name):
    def register(fn):
        LOADERS[name] = fn
        return fn
    return register


def read_frames(path: str, frames: list[int]) -> dict[int, np.ndarray]:
    """RGB frames by index."""
    cap, out, want = cv2.VideoCapture(path), {}, set(frames)
    for k in range(max(frames) + 1):
        ok, img = cap.read()
        if not ok:
            break
        if k in want:
            out[k] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def box_depth(depth: np.ndarray, box) -> float:
    """Median depth over the central half of the box (at least 3x3 pixels)."""
    h, w = depth.shape
    x0, y0, x1, y1 = box
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hw, hh = max(0.25 * (x1 - x0), 1.5), max(0.25 * (y1 - y0), 1.5)
    patch = depth[int(np.clip(cy - hh, 0, h - 1)):int(np.clip(cy + hh, 1, h)) + 1,
                  int(np.clip(cx - hw, 0, w - 1)):int(np.clip(cx + hw, 1, w)) + 1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size else float("nan")


def transformers_depth(weights: str, device: str, batch: int = 4):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(weights)
    net = AutoModelForDepthEstimation.from_pretrained(weights).to(device).eval()

    def predict(frames: list[np.ndarray], K: np.ndarray) -> list[np.ndarray]:
        out = []
        for i in range(0, len(frames), batch):
            chunk = frames[i:i + batch]
            sizes = [f.shape[:2] for f in chunk]
            inputs = proc(images=[Image.fromarray(f) for f in chunk], return_tensors="pt").to(device)
            with torch.no_grad():
                pred = net(**inputs)
            try:
                post = proc.post_process_depth_estimation(pred, target_sizes=sizes, source_sizes=sizes)
            except TypeError:  # processors without padding take only target sizes
                post = proc.post_process_depth_estimation(pred, target_sizes=sizes)
            out += [p["predicted_depth"].float().cpu().numpy() for p in post]
        return out
    return predict, False


@loader("depthpro")
def _depthpro(weights, device, code):
    return transformers_depth(weights, device, batch=2)


@loader("dav2_metric")
def _dav2(weights, device, code):
    return transformers_depth(weights, device)


@loader("zoedepth")
def _zoe(weights, device, code):
    return transformers_depth(weights, device)


@loader("moge2")
def _moge2(weights, device, code):
    import torch
    from moge.model.v2 import MoGeModel
    net = MoGeModel.from_pretrained(str(Path(weights) / "model.pt")).to(device).eval()

    def predict(frames, K):
        out = []
        for f in frames:
            fov_x = float(np.degrees(2 * np.arctan(f.shape[1] / (2 * K[0, 0]))))
            img = torch.tensor(f / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
            with torch.no_grad():
                res = net.infer(img, fov_x=fov_x)
            out.append(res["depth"].float().cpu().numpy())
        return out
    return predict, False


@loader("unidepth_v2")
def _unidepth(weights, device, code):
    import torch
    from unidepth.models import UniDepthV2
    net = UniDepthV2.from_pretrained(weights).to(device).eval()

    def predict(frames, K):
        out = []
        for f in frames:
            rgb = torch.from_numpy(f).permute(2, 0, 1)
            with torch.no_grad():
                res = net.infer(rgb, camera=torch.tensor(K, dtype=torch.float32))
            out.append(res["depth"][0, 0].float().cpu().numpy())
        return out
    return predict, False


@loader("vggt")
def _vggt(weights, device, code):
    import torch
    from vggt.models.vggt import VGGT
    net = VGGT.from_pretrained(weights).to(device).eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def predict(frames, K):  # all frames of one window together; depth up to one scale
        h, w = frames[0].shape[:2]
        nh = round(h * 518 / w / 14) * 14
        imgs = np.stack([cv2.resize(f, (518, nh), interpolation=cv2.INTER_CUBIC) for f in frames]).astype(np.float32) / 255.0
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2)[None].to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            pred = net(x)
        depth = pred["depth"][0, ..., 0].float().cpu().numpy()
        return [cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) for d in depth]
    return predict, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--model", required=True, choices=sorted(LOADERS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--code", nargs="*", default=[], help="source trees to put on sys.path (MoGe, UniDepth, vggt)")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    sys.path[:0] = args.code
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index, count = map(int, args.shard.split("/"))
    jobs = [json.loads(line) for i, line in enumerate(open(args.jobs)) if i % count == index]
    done = set()
    if Path(args.out).exists():
        done = {json.loads(line)["id"] for line in open(args.out)}
    predict, relative = LOADERS[args.model](args.weights, device, args.code)
    per_window = args.model == "vggt"
    with open(args.out, "a") as out:
        for job in jobs:
            windows = [w for w in job["windows"] if w["id"] not in done]
            if not windows:
                continue
            K = np.asarray(job["K"], dtype=float)
            frames = read_frames(job["video"], sorted({f for w in windows for f in w["frames"]}))
            cache = {}
            if not per_window:
                keys = sorted(frames)
                cache = dict(zip(keys, predict([frames[k] for k in keys], K)))
            for w in windows:
                ks = [k for k in w["frames"] if k in frames]
                maps = dict(zip(ks, predict([frames[k] for k in ks], K))) if per_window and len(ks) >= 2 else cache
                depth = [box_depth(maps[k], b) if k in maps else float("nan") for k, b in zip(w["frames"], w["boxes"])]
                out.write(json.dumps({"id": w["id"], "model": args.model, "frames": w["frames"], "depth": depth,
                                      "relative": relative}) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
