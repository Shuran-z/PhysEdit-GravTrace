"""Where does gravity show in each PhyEditing trajectory? Motion windows from the simulator states.

For every moving object, frames are labelled from the recorded velocities (30 fps):
  flight: acceleration equals the gravity vector (no contact force at all)
  slide:  horizontal deceleration along the velocity, no vertical acceleration, little spin
A window is a maximal run of one label. Output: one JSON line per window.

    python scripts/phyediting_windows.py COMPACT_DIR OUT.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FLIGHT_TOL = 0.04  # |a - g| below 4 % of g (plus 0.05 m/s^2)
MIN_SPEED = 0.03  # m/s


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of the True runs."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def windows(header: dict, z: np.lib.npyio.NpzFile) -> list[dict]:
    fps, g = float(header["fps"]), float(header["gravity"])
    gvec = np.asarray(header["gravity_vector"], dtype=float)
    out = []
    for j, obj in enumerate(header["objects"]):
        if obj.get("fixed"):
            continue
        v, w, p = z["velocity"][:, j], z["angular_velocity"][:, j], z["position"][:, j]
        if not np.isfinite(v).all():
            continue
        a = np.diff(v, axis=0) * fps  # a[k] between frames k and k+1
        speed = np.linalg.norm(v[1:], axis=1)
        moving = speed > MIN_SPEED
        flight = moving & (np.linalg.norm(a - gvec, axis=1) < FLIGHT_TOL * g + 0.05)
        vh = v[1:, :2]
        ah = a[:, :2]
        along = -(ah * vh).sum(1) / np.maximum(np.linalg.norm(vh, axis=1), 1e-9)  # deceleration along motion
        slide = (moving & ~flight & (np.abs(a[:, 2]) < 0.05 * g) & (along > 0.02 * g)
                 & (np.linalg.norm(w[1:], axis=1) < 1.0) & (np.linalg.norm(vh, axis=1) > MIN_SPEED))
        for label, mask in (("flight", flight), ("slide", slide)):
            for s, e in runs(mask):
                n = e - s + 1  # frames s..e inclusive bracket the run of e - s intervals
                if n < 3:
                    continue
                rec = {"sample_id": header["sample_id"], "event": header["event"], "background": header["background"],
                       "gravity": g, "object": obj["name"], "shape": obj.get("shape"), "label": label,
                       "start": int(s), "frames": int(n), "drop_m": float(p[s, 2] - p[e, 2]),
                       "speed0": float(np.linalg.norm(v[s])), "spin_max": float(np.linalg.norm(w[s:e + 1], axis=1).max())}
                if label == "slide":
                    rec["mu_eff"] = float(np.median(along[s:e]) / g)  # deceleration / g: the kinetic friction seen
                out.append(rec)
    return out


def main() -> None:
    compact, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    with open(out_path, "w") as f:
        for jpath in sorted(compact.glob("*.json")):
            header = json.loads(jpath.read_text())
            with np.load(jpath.with_suffix(".npz")) as z:
                for rec in windows(header, z):
                    f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
