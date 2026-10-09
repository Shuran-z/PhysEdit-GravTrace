"""Pinhole camera in the Blender/Kubric convention shared by all datasets used here.

`matrix_world` maps camera to world coordinates and the camera looks along its -Z axis.
`fx`, `fy` are focal lengths in units of image width and height, so a camera-frame point
(x, y, z) at depth d = -z lands on pixel u = (0.5 + fx x / d) W, v = (0.5 - fy y / d) H.
"""
from __future__ import annotations

import numpy as np


class Camera:
    def __init__(self, matrix_world, fx: float, fy: float, image_size) -> None:
        self.world_to_cam = np.linalg.inv(np.asarray(matrix_world, dtype=float))
        self.fx, self.fy = float(fx), abs(float(fy))
        self.width, self.height = (float(v) for v in image_size)

    def project(self, points: np.ndarray) -> np.ndarray:
        """(..., 3) world points -> (..., 2) pixels; NaN where a point is behind the camera."""
        cam = points @ self.world_to_cam[:3, :3].T + self.world_to_cam[:3, 3]
        depth = -cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (0.5 + self.fx * cam[..., 0] / depth) * self.width
            v = (0.5 - self.fy * cam[..., 1] / depth) * self.height
        uv = np.stack([u, v], axis=-1)
        uv[depth <= 1e-9] = np.nan
        return uv


def lookat_camera(position, look_at, fov_deg: float, image_size, up=(0.0, 0.0, 1.0)) -> dict:
    """`matrix_world`, `fx`, `fy` of a camera at `position` looking at `look_at` with vertical field of view
    `fov_deg` (the Genesis convention, as in PhyEditing)."""
    pos = np.asarray(position, dtype=float)
    forward = np.asarray(look_at, dtype=float) - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = right, np.cross(right, forward), -forward, pos
    fy = 0.5 / np.tan(np.radians(fov_deg) / 2.0)
    return {"matrix_world": matrix.tolist(), "fx": fy * image_size[1] / image_size[0], "fy": fy}


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix of an xyzw quaternion (identity for None)."""
    if q is None:
        return np.eye(3)
    x, y, z, w = np.asarray(q, dtype=float)[:4] / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
