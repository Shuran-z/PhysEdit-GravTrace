"""Baseline gravity estimates for the same windows and tracks GravTrace sees.

  pixel2d_size    image-plane parabola of the box centre; metres per pixel from the object's known size
  pixel2d_depth   the same parabola; metres per pixel from the declared initial depth (Z0 / f)
  pixel2d_matched the information GravTrace gets (camera, declared position and velocity, gravity direction) with a
                  2D model: the image track is the declared flight under the projection linearised at the start
  lift_oracle     box centre lifted to 3D with the true depth of the object centre (upper bound of lifting)
  lift_<model>_raw       box centre lifted with a depth model's metric depth
  lift_<model>_anchored  the same, with the model's depth rescaled so the first frame matches the declared state
  ..._v0                 lifted tracks fitted with the declared initial velocity (only g and an offset free)

Lifted tracks are fitted with a parabola per axis; g is the fitted acceleration along the known gravity
direction (the camera pose is known, as it is to GravTrace). Output: one predictions file per method, in the
format `python -m gravtrace score` reads.

    python scripts/baselines.py SAMPLES.jsonl OUT_DIR [--depth DEPTH.jsonl ...] [--compact DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gravtrace.camera import quat_to_matrix  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0])  # Blender camera (y up, looks along -z) -> OpenCV camera (y down, looks along +z)


def camera_frame(sample: dict):
    """World->OpenCV-camera rotation and translation, pixel intrinsics."""
    m = np.asarray(sample["camera"]["matrix_world"], dtype=float)
    w2c = np.linalg.inv(m)
    width, height = sample["image_size"]
    cam = sample["camera"]
    K = np.array([[cam["fx"] * width, 0, cam.get("cx", 0.5) * width], [0, abs(cam["fy"]) * height, cam.get("cy", 0.5) * height], [0, 0, 1]])
    return FLIP @ w2c[:3, :3], FLIP @ w2c[:3, 3], K


def surface_depth(sample: dict, R: np.ndarray, T: np.ndarray) -> float:
    """z-depth (OpenCV camera) where the ray through the object's centre enters its bounding box at the window start."""
    obj = sample["object"]
    corners = np.asarray(obj["corners"], dtype=float)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    rot = quat_to_matrix(obj.get("quaternion"))
    centre = np.asarray(obj["position"], dtype=float)
    cam = -R.T @ T  # camera centre, world
    ray = (centre - cam) / np.linalg.norm(centre - cam)
    o, d = rot.T @ (cam - centre), rot.T @ ray
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    t_entry = np.nanmax(np.minimum(t1, t2))
    return float((R @ (cam + t_entry * ray) + T)[2])


def projection_jacobian(p_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """d(pixel)/d(camera-frame point) of the pinhole projection at p_cam (OpenCV camera)."""
    x, y, z = p_cam
    return np.array([[K[0, 0] / z, 0.0, -K[0, 0] * x / z ** 2], [0.0, K[1, 1] / z, -K[1, 1] * y / z ** 2]])


def quadratic_accel(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares y = c0 + c1 t + a t^2 / 2 per column; returns a."""
    A = np.column_stack([np.ones_like(t), t, 0.5 * t * t])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[2]


def methods_for(sample: dict, depths: dict[str, dict], true_centre_depth) -> dict[str, float | None]:
    frames = np.asarray(sample["boxes"]["frames"], dtype=int)
    boxes = np.asarray(sample["boxes"]["xyxy"], dtype=float).reshape(-1, 4)
    seen = sample["window"].get("edges_visible", {})  # the same visibility GravTrace gets: a box centre needs all 4 edges
    full = np.array([all(seen.get(str(int(f)), [True] * 4)) for f in frames], dtype=bool)
    frames, boxes = frames[full], boxes[full]
    out: dict[str, float | None] = {}
    if len(frames) < 3:
        return out
    t = (frames - sample["window"]["start_frame"]) / float(sample["fps"])
    uv = 0.5 * (boxes[:, :2] + boxes[:, 2:])
    R, T, K = camera_frame(sample)
    g_cam = R @ np.asarray(sample["motion"]["gravity_dir"], dtype=float)
    centre0 = R @ np.asarray(sample["object"]["position"], dtype=float) + T
    a_px = float(np.linalg.norm(quadratic_accel(t, uv)))
    corners = np.asarray(sample["object"]["corners"], dtype=float)
    size = float((corners.max(axis=0) - corners.min(axis=0)).max())
    extent_px = np.median(np.maximum(boxes[:3, 2] - boxes[:3, 0], boxes[:3, 3] - boxes[:3, 1]))
    out["pixel2d_size"] = a_px * size / extent_px if extent_px > 0 else None
    out["pixel2d_depth"] = a_px * centre0[2] / (0.5 * (K[0, 0] + K[1, 1]))
    # Unknown velocity: free image-plane linear terms, scalar acceleration
    # constrained to the projected declared gravity direction (no reference v0).
    J = projection_jacobian(centre0, K) @ R
    g_px = J @ np.asarray(sample["motion"]["gravity_dir"], dtype=float)
    A = np.zeros((uv.size, 5))
    A[0::2, 0], A[1::2, 1] = 1., 1.
    A[0::2, 2], A[1::2, 3] = t, t
    A[:, 4] = np.outer(.5*t*t, g_px).ravel()
    out["pixel2d_agnostic"] = float(np.linalg.lstsq(A, uv.ravel(), rcond=None)[0][4])
    if "v0" in sample["motion"]:  # same information as GravTrace, 2D model: projection linearised at the declared position
        J = projection_jacobian(centre0, K) @ R  # pixels per metre of world displacement
        v_px, g_px = J @ np.asarray(sample["motion"]["v0"], dtype=float), J @ np.asarray(sample["motion"]["gravity_dir"], dtype=float)
        y = (uv - np.outer(t, v_px)).ravel()  # = c + g * g_px t^2 / 2
        A = np.zeros((y.size, 3))
        A[0::2, 0], A[1::2, 1] = 1.0, 1.0
        A[:, 2] = np.outer(0.5 * t * t, g_px).ravel()
        out["pixel2d_matched"] = float(np.linalg.lstsq(A, y, rcond=None)[0][2])

    v_cam = R @ np.asarray(sample["motion"]["v0"], dtype=float) if "v0" in sample["motion"] else None

    def lift(depth: np.ndarray, declared: bool = False) -> float | None:
        """g along the known gravity direction from the lifted track; `declared` fixes the track's initial velocity
        to the declared one (its position offset stays free, as tracks and declarations differ by a constant)."""
        ok = np.isfinite(depth) & (depth > 0)
        if sample["window"].get("depth_require_all_frames", False) and not ok.all():
            return None
        if ok.sum() < 3:
            return None
        rays = np.column_stack([(uv[ok, 0] - K[0, 2]) / K[0, 0], (uv[ok, 1] - K[1, 2]) / K[1, 1], np.ones(ok.sum())])
        pts, tt = rays * depth[ok, None], t[ok]
        if not declared:
            return float(quadratic_accel(tt, pts) @ g_cam)
        y = (pts - np.outer(tt, v_cam)) @ g_cam  # along gravity: c0 + g tt^2 / 2
        A = np.column_stack([np.ones_like(tt), 0.5 * tt * tt])
        return float(np.linalg.lstsq(A, y, rcond=None)[0][1])

    if true_centre_depth is not None:
        out["lift_oracle"] = lift(true_centre_depth(frames))
        if v_cam is not None:
            out["lift_oracle_v0"] = lift(true_centre_depth(frames), declared=True)
    z_surface0 = surface_depth(sample, R, T)
    for model, rec in depths.items():
        by_frame = dict(zip(rec["frames"], rec["depth"]))
        d = np.array([by_frame.get(int(k), np.nan) for k in frames], dtype=float)
        if not rec.get("relative"):
            out[f"lift_{model}_raw"] = lift(d)
        first = d[np.isfinite(d) & (d > 0)]
        anchor_depth = first[0] if first.size else None
        if sample["window"].get("depth_anchor_extrapolate", False):
            finite = np.isfinite(d) & (d > 0)
            if finite.sum() >= 3:
                design = np.column_stack([np.ones(finite.sum()), t[finite], .5*t[finite]**2])
                anchor_depth = float(np.linalg.lstsq(design, d[finite], rcond=None)[0][0])
            else:
                anchor_depth = None
        valid_anchor = anchor_depth is not None and anchor_depth > 0
        out[f"lift_{model}_anchored"] = lift(d * z_surface0 / anchor_depth) if valid_anchor else None
        if v_cam is not None and first.size:
            out[f"lift_{model}_anchored_v0"] = lift(d * z_surface0 / first[0], declared=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("out", type=Path)
    ap.add_argument("--depth", nargs="*", default=[])
    ap.add_argument("--compact", type=Path, default=Path("data/compact"), help="true states, for lift_oracle")
    args = ap.parse_args()
    depths: dict[str, dict] = defaultdict(dict)
    for path in args.depth:
        for line in open(path):
            rec = json.loads(line)
            depths[rec["id"]][rec["model"]] = rec
    preds: dict[str, list] = defaultdict(list)
    states: dict = {}
    for line in open(args.samples):
        s = json.loads(line)
        true_depth = None
        if args.compact:
            if s["gt_id"] not in states:
                states.clear()
                z = np.load(args.compact / f"{s['gt_id']}.npz")
                states[s["gt_id"]] = (z["position"], [str(n) for n in z["names"]])
            pos, names = states[s["gt_id"]]
            j = names.index(s["id"].split("__")[-2])
            R, T, _ = camera_frame(s)
            true_depth = lambda fr, pos=pos, j=j, R=R, T=T: (pos[fr, j] @ R.T + T)[:, 2]
        for method, g in methods_for(s, depths.get(s["id"], {}), true_depth).items():
            ok = g is not None and np.isfinite(g) and g > 0
            preds[method].append({"id": s["id"], "status": "ok" if ok else "failed", **({"gravity": float(g)} if ok else {})})
    args.out.mkdir(parents=True, exist_ok=True)
    for method, rows in preds.items():
        with open(args.out / f"{method}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    print({m: len(r) for m, r in preds.items()})


if __name__ == "__main__":
    main()
