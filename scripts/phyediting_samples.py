"""GravTrace samples from PhyEditing free-flight windows.

Each window (from phyediting_windows.py) declares the target object, the first frame of the flight
and its length; the object's state at that frame (position, orientation, velocity) is the declared
initial state, as the conditioning frame of a generated video would provide.

    python scripts/phyediting_samples.py COMPACT_DIR WINDOWS.jsonl OUT.jsonl [--camera cam01]
        [--obs oracle] [--v0 declared|free] [--min-frames 6]

`--obs oracle` observes the box spanned by the true object corners (true rotation included), which
isolates model error from tracking error; otherwise the sample points at masks to be filled later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gravtrace.camera import Camera, lookat_camera, quat_to_matrix  # noqa: E402


MESH_ROOT = Path(__file__).resolve().parents[1] / "third_party/meshes"  # copies of the rendered (visual) meshes
_MESHES: dict = {}


def mesh_outline(obj: dict) -> np.ndarray | None:
    """Convex-hull vertices (object frame) of the object's visual mesh, when a copy is available and its extent matches
    the declared size (same frame and scale); else None. The silhouette a tracker sees is the mesh's, not the proxy's."""
    from scipy.spatial import ConvexHull
    path = obj.get("visual_mesh_path")
    if not path or obj.get("scale"):
        return None
    if path not in _MESHES:
        local = MESH_ROOT / path.lstrip("/")
        if not local.exists():  # the same asset under another release's directory
            same = sorted(MESH_ROOT.rglob(Path(path).name)) if MESH_ROOT.exists() else []
            local = same[0] if same else None
        pts = None
        if local is not None:
            v = np.array([[float(x) for x in line.split()[1:4]] for line in open(local, errors="ignore") if line.startswith("v ")])
            if len(v) >= 4:
                pts = v[ConvexHull(v).vertices]
                if len(pts) > 400:
                    pts = pts[np.linspace(0, len(pts) - 1, 400).astype(int)]
        _MESHES[path] = pts
    pts = _MESHES[path]
    size = obj.get("collision_proxy_size") or obj.get("size")
    if pts is None:
        return None
    if not size:  # nothing to check the copy against beyond its existence
        return pts
    extent = pts.max(axis=0) - pts.min(axis=0)
    return pts if np.allclose(extent, size, rtol=0.03, atol=0.002) else None


def geometry(obj: dict) -> tuple[np.ndarray, list[float]]:
    """Outline points (object frame) and principal moments of inertia (kg m^2). The outline is the visual mesh's hull
    when available, else the collision proxy's (a box, or a z-axis cylinder sampled on its two rims); the inertia is
    always the proxy's, as in the simulation."""
    mesh = mesh_outline(obj)
    if not (obj.get("collision_proxy_size") or obj.get("size")):  # undeclared size: the mesh's extent, if any
        if mesh is None:
            return np.zeros((1, 3)), [1e-6, 1e-6, 1e-6]
        obj = {**obj, "size": (mesh.max(axis=0) - mesh.min(axis=0)).tolist()}
    points, inertia = proxy_geometry(obj)
    return (mesh if mesh is not None else points), inertia


def proxy_geometry(obj: dict) -> tuple[np.ndarray, list[float]]:
    """Outline points and principal moments of inertia of the collision proxy."""
    size = np.asarray(obj.get("collision_proxy_size") or obj["size"], dtype=float)
    m = float(obj.get("mass") or 1.0)
    if obj.get("collision_proxy") == "cylinder":
        r, h = size[0] / 2.0, size[2]
        a = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        pts = np.array([[r * np.cos(t), r * np.sin(t), z] for z in (-h / 2, h / 2) for t in a])
        return pts, [m * (3 * r * r + h * h) / 12, m * (3 * r * r + h * h) / 12, m * r * r / 2]
    hs = size / 2.0
    pts = np.array([[x, y, z] for x in (-hs[0], hs[0]) for y in (-hs[1], hs[1]) for z in (-hs[2], hs[2])])
    return pts, [m * (size[1] ** 2 + size[2] ** 2) / 12, m * (size[0] ** 2 + size[2] ** 2) / 12, m * (size[0] ** 2 + size[1] ** 2) / 12]


def surface_points(obj: dict) -> np.ndarray:
    """Points on the object's surface (object frame): the visual mesh hull, cylinder rims, or box corners, edge
    midpoints and face centres."""
    pts, _ = geometry(obj)
    if mesh_outline(obj) is not None or obj.get("collision_proxy") == "cylinder":
        return pts
    h = np.abs(pts).max(axis=0)
    grid = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)])
    return grid * h


def occluded(cam_centre: np.ndarray, points: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    """For each world point, whether the segment from the camera to it passes through any of the oriented boxes
    (centre, rotation, half-extents) before reaching it."""
    hidden = np.zeros(len(points), dtype=bool)
    d = points - cam_centre
    for centre, rot, half in boxes:
        o, dd = (cam_centre - centre) @ rot, d @ rot  # in the box frame
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - o) / dd, (half - o) / dd
        t_in = np.nanmax(np.minimum(t1, t2), axis=1)
        t_out = np.nanmin(np.maximum(t1, t2), axis=1)
        hidden |= (t_in < t_out) & (t_out > 0) & (t_in < 0.995)
    return hidden


def edge_visibility(header: dict, z, j: int, frames: np.ndarray, camera: Camera) -> list[list[bool]]:
    """Per frame, whether the left/top/right/bottom edge of the target's image box is seen, i.e. some surface point that
    attains it is not hidden behind another declared object (static fixtures or the other moving objects)."""
    names = [str(n) for n in z["names"]]
    target = next(o for o in header["objects"] if o["name"] == names[j])
    surf = surface_points(target)
    cam_centre = np.asarray(camera.world_to_cam, dtype=float)
    cam_centre = -np.linalg.inv(cam_centre[:3, :3]) @ cam_centre[:3, 3]
    out = []
    for k in frames:
        others = []
        for i, o in enumerate(header["objects"]):
            if i == j or names[i] != o["name"]:
                continue
            size = np.asarray(o.get("collision_proxy_size") or o.get("size") or [0, 0, 0], dtype=float)
            if size.min() <= 0 or not np.isfinite(z["position"][k, i]).all():
                continue
            others.append((z["position"][k, i], quat_to_matrix(z["quaternion_xyzw"][k, i]), size / 2.0))
        pts = z["position"][k, j] + surf @ quat_to_matrix(z["quaternion_xyzw"][k, j]).T
        uv = camera.project(pts)
        hidden = occluded(cam_centre, pts, others)
        vis = []
        for axis, sign in ((0, -1), (1, -1), (0, 1), (1, 1)):  # left, top, right, bottom
            value = sign * uv[:, axis]
            extreme = value >= np.nanmax(value) - 1.5
            vis.append(bool((extreme & ~hidden).any()))
        out.append(vis)
    return out


def make_sample(header: dict, z, win: dict, camera_id: str, obs: str, v0_mode: str) -> dict:
    names = [str(n) for n in z["names"]]
    j = names.index(win["object"])
    obj = next(o for o in header["objects"] if o["name"] == win["object"])
    s, n = win["start"], win["frames"]
    width, height = header["resolution"]
    view = header["camera"][camera_id]
    cam = lookat_camera(view["position"], view["look_at"], view["fov_deg"], (width, height),
                        principal_offset_px=view.get("principal_offset_px", (0.0, 0.0)))
    corners, inertia = geometry(obj)
    gdir = np.asarray(header["gravity_vector"], dtype=float)
    v0 = z["velocity"][s, j]
    sample = {
        "id": f"{header['sample_id']}__{camera_id}__{win['object']}__f{s}",
        "gt_id": header["sample_id"], "source": header["event"], "scenario": "projectile",
        "fps": header["fps"], "image_size": [width, height], "camera": cam,
        "object": {"corners": corners.tolist(), "position": z["position"][s, j].tolist(),
                   "quaternion": z["quaternion_xyzw"][s, j].tolist(), "angular_velocity": z["angular_velocity"][s, j].tolist(),
                   "mass": obj.get("mass"), "inertia": inertia, "linear_damping": obj.get("linear_damping") or 0.0,
                   "angular_damping": obj.get("angular_damping") or 0.0},
        "motion": {"gravity_dir": (gdir / np.linalg.norm(gdir)).tolist(), "t0": [0.0, 0.0]},
        "window": {"start_frame": s, "max_frames": n},
        "truth": {"gravity": header["gravity"]},
        "meta": {"event": header["event"], "background": header["background"], "camera": camera_id,
                 "gravity": header["gravity"], "drop_m": win["drop_m"], "frames": n, "spin_max": win["spin_max"]},
    }
    if v0_mode == "declared":
        sample["motion"]["v0"] = v0.tolist()
    else:  # unknown launch: speed within +-50 % of the declared one, elevation and azimuth free
        speed = float(np.linalg.norm(v0))
        sample["motion"].update(speed=[0.5 * speed, 1.5 * speed + 0.05], angle_deg=[-89.0, 89.0],
                                directions=[(v0 / max(speed, 1e-9)).tolist()])
    if obs == "oracle":
        camera = Camera.from_dict(cam, (width, height))
        frames = np.arange(s, s + n)
        pts = np.stack([z["position"][k, j] + corners @ quat_to_matrix(z["quaternion_xyzw"][k, j]).T for k in frames])
        uv = camera.project(pts)
        full = np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1)
        boxes = np.concatenate([np.clip(full[:, :2], 0, None), np.minimum(full[:, 2:], [width, height])], axis=1)
        area = lambda b: np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
        visible = np.isfinite(full).all(axis=1) & (area(boxes) >= 0.5 * area(full))  # at least half the box in view
        sample["boxes"] = {"frames": frames[visible].tolist(), "xyxy": boxes[visible].tolist()}
        sample["meta"]["visible_frames"] = int(visible.sum())
    # which box edges the declared scene leaves visible in each frame of the window (occlusion by other objects)
    frames_all = np.arange(s, s + n)
    camera = Camera.from_dict(cam, (width, height))
    sample["window"]["edges_visible"] = dict(zip(map(str, frames_all.tolist()), edge_visibility(header, z, j, frames_all, camera)))
    return sample


def prompt_boxes(header: dict, z, camera_id: str, frame: int = 0) -> dict:
    """Projected box of every movable object in the prompt frame (the declared state), if mostly in view."""
    width, height = header["resolution"]
    view = header["camera"][camera_id]
    cam = lookat_camera(view["position"], view["look_at"], view["fov_deg"], (width, height),
                        principal_offset_px=view.get("principal_offset_px", (0.0, 0.0)))
    camera = Camera.from_dict(cam, (width, height))
    names = [str(n) for n in z["names"]]
    out = {}
    for obj in header["objects"]:
        if obj.get("fixed"):
            continue
        j = names.index(obj["name"])
        pts = z["position"][frame, j] + geometry(obj)[0] @ quat_to_matrix(z["quaternion_xyzw"][frame, j]).T
        uv = camera.project(pts)
        if not np.isfinite(uv).all():
            continue
        x0, y0 = np.clip(uv.min(axis=0), 0, [width, height])
        x1, y1 = np.clip(uv.max(axis=0), 0, [width, height])
        full = np.prod(uv.max(axis=0) - uv.min(axis=0))
        if (x1 - x0) * (y1 - y0) >= 0.5 * full and min(x1 - x0, y1 - y0) >= 4:
            out[obj["name"]] = [float(x0), float(y0), float(x1), float(y1)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("compact", type=Path)
    ap.add_argument("windows", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--camera", default="cam01")
    ap.add_argument("--obs", default="oracle")
    ap.add_argument("--v0", default="declared", choices=["declared", "free"])
    ap.add_argument("--min-frames", type=int, default=6)
    ap.add_argument("--tracks", type=Path, help="observe SAM2 tracks (track_sam2.py output) instead of the oracle")
    ap.add_argument("--jobs", type=Path, help="also write SAM2 tracking jobs for the samples written")
    ap.add_argument("--video-root", default="data/phyediting", help="video path prefix used in the jobs")
    ap.add_argument("--only", type=Path, help="restrict to the sample ids listed in this file")
    args = ap.parse_args()
    only = set(args.only.read_text().split()) if args.only else None
    tracks = {}
    if args.tracks:
        for line in open(args.tracks):
            t = json.loads(line)
            tracks[t["id"]] = t
    jobs: dict = {}
    cache: dict = {}
    with open(args.out, "w") as f:
        for line in open(args.windows):
            win = json.loads(line)
            if win["label"] != "flight" or win["frames"] < args.min_frames:
                continue
            sid = win["sample_id"]
            if only is not None and sid not in only:
                continue
            if sid not in cache:
                cache.clear()
                cache[sid] = (json.loads((args.compact / f"{sid}.json").read_text()), np.load(args.compact / f"{sid}.npz"))
            header, z = cache[sid]
            if not header["camera"].get(args.camera) or not header["videos"].get(args.camera):
                continue  # no recorded view or no video for this camera
            sample = make_sample(header, z, win, args.camera, args.obs, args.v0)
            job_id = f"{sid}__{args.camera}"
            if args.tracks:
                track = tracks.get(job_id, {}).get("tracks", {}).get(win["object"])
                frames = set(range(win["start"], win["start"] + win["frames"]))
                keep = [(fr, b) for fr, b in zip(track["frames"], track["xyxy"]) if fr in frames] if track else []
                sample["boxes"] = {"frames": [k for k, _ in keep], "xyxy": [b for _, b in keep]}
            if args.jobs and job_id not in jobs:
                jobs[job_id] = {"id": job_id, "video": f"{args.video_root}/{header['videos'][args.camera]}", "prompt_frame": 0,
                                "objects": prompt_boxes(header, z, args.camera), "last_frame": 0}
            if args.jobs:
                jobs[job_id]["last_frame"] = max(jobs[job_id]["last_frame"], win["start"] + win["frames"] + 1)
            f.write(json.dumps(sample) + "\n")
    if args.jobs:
        with open(args.jobs, "w") as f:
            for job in jobs.values():
                f.write(json.dumps(job) + "\n")


if __name__ == "__main__":
    main()
