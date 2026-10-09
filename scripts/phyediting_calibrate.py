"""Check (and correct) the image offset between rendered videos and their recorded cameras.

For a trajectory, the moving objects' projected outlines (recorded 3D states, recorded camera) are compared with the
pixels that change between the static condition frame and later frames. The image shift that best aligns them
should be zero. Some releases are offset (T08: about 40 px down in every camera), which this measures per
(source, background, camera) as the median over several trajectories and, with --apply, records in the compact
headers as `principal_offset_px` so every projection uses the corrected principal point.

    python scripts/phyediting_calibrate.py COMPACT_DIR --root data/phyediting --prefix T08V4F480 [--per-group 12] [--apply]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gravtrace.camera import Camera, lookat_camera, quat_to_matrix  # noqa: E402
from phyediting_samples import geometry  # noqa: E402

FRAMES = (40, 60, 80, 100, 120)


def offset(header: dict, z, root: Path, cam_id: str) -> tuple[float, float, float] | None:
    """(dx, dy, IoU at the best shift) for one video, or None when nothing moves enough to tell."""
    view = header["camera"].get(cam_id)
    if not view or cam_id not in header["videos"]:
        return None
    cap = cv2.VideoCapture(str(root / header["videos"][cam_id]))
    width, height = int(cap.get(3)), int(cap.get(4))
    cam = Camera.from_dict(lookat_camera(view["position"], view["look_at"], view["fov_deg"], (width, height)), (width, height))
    names = [str(n) for n in z["names"]]
    gray = {}
    for k in (23,) + FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, img = cap.read()
        if ok:
            gray[k] = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
    outlines = {}
    for k in gray:
        polys = []
        for o in header["objects"]:
            if o.get("fixed"):
                continue
            j = names.index(o["name"])
            uv = cam.project(z["position"][k, j] + geometry(o)[0] @ quat_to_matrix(z["quaternion_xyzw"][k, j]).T)
            if np.isfinite(uv).all():
                polys.append(cv2.convexHull(uv.astype(np.float32)))
        outlines[k] = polys
    pairs = []
    for k in FRAMES:
        if k not in gray:
            continue
        motion = (np.abs(gray[k] - gray[23]) > 12).astype(np.uint8)
        if motion.sum() > 30:
            pairs.append((k, motion))
    if not pairs:
        return None

    def score(dx: int, dy: int) -> float:
        total = 0.0
        for k, motion in pairs:
            mask = np.zeros((height, width), np.uint8)
            for kk in (23, k):
                for poly in outlines[kk]:
                    cv2.fillConvexPoly(mask, (poly + [dx, dy]).astype(np.int32), 1)
            inter = float((mask & motion).sum())
            total += inter / (mask.sum() + motion.sum() - inter + 1e-9)
        return total / len(pairs)

    best = max(((score(dx, dy), dx, dy) for dx in range(-80, 81, 8) for dy in range(-80, 81, 8)))
    for step in (4, 2, 1):
        _, bx, by = best
        best = max(best, *((score(bx + i * step, by + j * step), bx + i * step, by + j * step) for i in (-1, 0, 1) for j in (-1, 0, 1)))
    return float(best[1]), float(best[2]), float(best[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("compact", type=Path)
    ap.add_argument("--root", type=Path, default=Path("data/phyediting"))
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--per-group", type=int, default=12)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-shift", type=float, default=4.0, help="record an offset only when its size exceeds this (px)")
    args = ap.parse_args()
    groups = defaultdict(list)
    for path in sorted(args.compact.glob(f"{args.prefix}*.json")):
        h = json.loads(path.read_text())
        for cam in ("cam01", "cam02", "cam03"):
            if h["camera"].get(cam):
                groups[(h["background"], cam)].append(path)
    result = {}
    for (bg, cam), paths in sorted(groups.items()):
        picks = paths[:: max(1, len(paths) // args.per_group)][: args.per_group]
        found = []
        for path in picks:
            with np.load(path.with_suffix(".npz")) as z:
                r = offset(json.loads(path.read_text()), z, args.root, cam)
            if r and r[2] > 0.2:  # a clear match
                found.append(r)
        if not found:
            print(bg, cam, "no clear match")
            continue
        dx, dy = np.median([f[0] for f in found]), np.median([f[1] for f in found])
        spread = np.median([abs(f[1] - dy) + abs(f[0] - dx) for f in found])
        result[(bg, cam)] = (float(dx), float(dy))
        print(f"{bg} {cam}: offset ({dx:+.0f}, {dy:+.0f}) px from {len(found)} videos, median deviation {spread:.1f} px")
    if args.apply:
        for (bg, cam), (dx, dy) in result.items():
            if np.hypot(dx, dy) < args.min_shift:
                continue
            for path in groups[(bg, cam)]:
                h = json.loads(path.read_text())
                h["camera"][cam]["principal_offset_px"] = [dx, dy]
                note = f"{cam} image offset ({dx:+.0f}, {dy:+.0f}) px from the recorded camera, calibrated"
                h["camera_issues"] = [i for i in h.get("camera_issues", []) if not i.startswith(f"{cam} image offset")] + [note]
                path.write_text(json.dumps(h))
        print("applied")


if __name__ == "__main__":
    main()
