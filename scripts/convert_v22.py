"""Convert a GravTrace v22 manifest (PISA / NewtonBench / self-rendered scenes) to GravTrace samples.

Every quantity written here is known before the video is generated: camera, object geometry and
initial pose, gravity direction, the declared initial velocity or its range, and the time of the
first observed frame. Gravity magnitude goes only into `truth`.

usage: convert_v22.py V22.jsonl OUT.jsonl
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gravtrace.camera import quat_to_matrix  # noqa: E402

DEFAULT_MASK_PATTERN = "segmentation_*.npy"


def box_corners(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=float)


def gso_obj_lines(target: dict) -> list[str]:
    """Lines of a Google Scanned Objects mesh, from the extracted cache or its archive."""
    cache = Path(target["gso_cache_dir"])
    obj = cache / target["asset_id"] / "visual_geometry.obj"
    if obj.exists():
        return open(obj, errors="ignore").read().splitlines()
    with tarfile.open(cache / f"{target['asset_id']}.tar.gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("visual_geometry.obj"))
        return tar.extractfile(member).read().decode(errors="ignore").splitlines()


def object_corners(target: dict) -> np.ndarray:
    """Axis-aligned box of the object's geometry in its own frame, scale applied."""
    if target.get("mesh_kind") == "primitive":
        shape, geo = target["shape"], target["geometry"]
        half = {"sphere": lambda: [geo["radius"]] * 3,
                "box": lambda: [v / 2 for v in geo["size"][:3]],
                "cylinder": lambda: [geo["radius"], geo["radius"], geo["height"] / 2]}[shape]()
    else:
        v = np.array([line.split()[1:4] for line in gso_obj_lines(target) if line.startswith("v ")], dtype=float)
        return box_corners(v.min(axis=0), v.max(axis=0)) * np.resize(target.get("scale", [1.0]), 3)
    half = np.array(half, dtype=float) * np.resize(target.get("scale", [1.0]), 3)
    return box_corners(-half, half)


def declared_target(row: dict) -> tuple[dict, str]:
    """The candidate whose object index matches the declared segmentation label.

    Dynamic objects take the highest label ids, in object order, so index = label - base where
    base is the smallest of the last `len(candidates)` labels present in the masks.
    """
    candidates = [row["target"]] + list(row["target"].get("alternates") or [])
    obs = row["observations"]
    if obs.get("target_label") is None:
        return candidates[0], "only_candidate" if len(candidates) == 1 else "first_candidate"
    labels = set()
    for path in sorted(Path(obs["mask_dir"]).glob(obs.get("mask_pattern", DEFAULT_MASK_PATTERN)))[:64]:
        raw = np.squeeze(np.load(path)) if path.suffix == ".npy" else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        labels.update(int(v) for v in np.unique(raw) if v)
    ordered = sorted(labels)
    if len(ordered) >= len(candidates):
        index = int(obs["target_label"]) - ordered[-len(candidates)]
        for c in candidates:
            if c.get("object_index") == index:
                return c, "declared_label"
    return candidates[0], "fallback_first_candidate"


def first_frame_time(row: dict, variant: dict) -> list[float]:
    """Time of the first observed frame after the declared initial state, as a [lo, hi] range.

    Self-rendered scenes declare it exactly: the body is released in the declared state at the
    dynamics anchor frame, where the fit window starts.
    """
    if ((row.get("initial_contract") or {}).get("rigid_rollout") or {}).get("timing"):
        return [0.0, 0.0]
    return [float(v) for v in variant["time_offset_bounds_s"]]


def motion(row: dict, target: dict, corners: np.ndarray) -> dict:
    variants = row["motion_variants"]
    first = variants[0]
    rigid = (row.get("initial_contract") or {}).get("rigid_rollout") or {}
    m = {"gravity_dir": first.get("gravity_axis") or rigid.get("gravity_direction_world") or [0.0, 0.0, -1.0],
         "t0": first_frame_time(row, first)}
    if row["scenario"] == "incline":
        slopes = {}
        for v in variants:  # variants differ in ramp azimuth and in the start point's position on the ramp
            s = v["incline"]
            key = json.dumps([s["downslope_dir"], s["angle_deg"], s["mu"], bool(s.get("rolling"))])
            slope = slopes.setdefault(key, {"dir": s["downslope_dir"], "angle_deg": s["angle_deg"], "mu": s["mu"],
                                            "rolling": bool(s.get("rolling")), "length": 0.0})
            slope["length"] = max(slope["length"], float(s.get("length_m") or 0.0))  # full ramp length
        m.update(slopes=list(slopes.values()), speed=first.get("speed_bounds", [0.0, 0.0]))
    elif first.get("v0_known") is not None:
        m["v0"] = first["v0_known"]
    elif row["scenario"] == "projectile":
        directions = []
        for v in variants:
            if v["v0_direction"] not in directions:
                directions.append(v["v0_direction"])
        m.update(speed=first["speed_bounds"], angle_deg=first["launch_angle_bounds_deg"], directions=directions)
    # contact ends the analytic flight: keep the highest declared floor below the object's lowest corner
    lowest = (corners @ quat_to_matrix(target.get("quaternion0_xyzw")).T + target["position0"])[:, 2].min()
    floors = [v["ground_z"] for v in variants if v.get("ground_z") is not None and v["ground_z"] < lowest - 1e-3]
    if floors:
        m["ground_z"] = max(floors)
    return m


def convert(row: dict) -> tuple[dict, str]:
    target, how = declared_target(row)
    try:
        corners = object_corners(target)
    except (FileNotFoundError, StopIteration):  # mesh not available: the fit falls back to the box centre
        corners, how = None, how + "+no_mesh"
    intr = row["camera"]["intrinsics"]
    sample = {
        "id": row["sample_id"], "source": row["source"], "scenario": row["scenario"],
        "fps": row["video"]["fps"], "image_size": row["video"]["image_size"],
        "camera": {"matrix_world": row["camera"]["matrix_world"], "fx": intr[0][0], "fy": abs(intr[1][1])},
        "object": {"position": target["position0"], "quaternion": target.get("quaternion0_xyzw") or [0.0, 0.0, 0.0, 1.0]},
        "motion": motion(row, target, np.zeros((1, 3)) if corners is None else corners),
        "masks": {"dir": row["observations"]["mask_dir"],
                  "pattern": row["observations"].get("mask_pattern", DEFAULT_MASK_PATTERN),
                  "label": row["observations"].get("target_label")},
        "truth": {"gravity": row["truth"]["gravity_m_s2"]},
    }
    if corners is not None:
        sample["object"]["corners"] = corners.round(6).tolist()
    contract = row.get("initial_contract") or {}
    timing = (contract.get("rigid_rollout") or {}).get("timing") or {}
    window = {}
    if timing.get("dynamics_anchor_frame") is not None:  # rendered scenes hold still until the anchor frame
        window["start_frame"] = int(timing["dynamics_anchor_frame"])
    if contract.get("projection_fit_max_frames"):  # frames of free motion after the anchor
        window["max_frames"] = int(contract["projection_fit_max_frames"]) + ("start_frame" in window)
    if window:
        sample["window"] = window
    return sample, how


def main() -> None:
    source, dest = sys.argv[1:3]
    counts: dict[str, int] = {}
    with open(dest, "w", encoding="utf-8") as out:
        for line in open(source, encoding="utf-8"):
            sample, how = convert(json.loads(line))
            counts[how] = counts.get(how, 0) + 1
            out.write(json.dumps(sample) + "\n")
    print("target resolution:", counts)


if __name__ == "__main__":
    main()
