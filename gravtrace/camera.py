"""Pinhole camera in the Blender/Kubric convention shared by all datasets used here.

`matrix_world` maps camera to world coordinates and the camera looks along its -Z axis.
`fx`, `fy` are focal lengths in units of image width and height, so a camera-frame point
(x, y, z) at depth d = -z lands on pixel u = (cx + fx x / d) W, v = (cy - fy y / d) H, with the principal point
(cx, cy) at the image centre (0.5, 0.5) unless the camera declares otherwise.
"""
from __future__ import annotations

import numpy as np


class Camera:
    def __init__(self, matrix_world, fx: float, fy: float, image_size, cx: float = 0.5, cy: float = 0.5) -> None:
        self.world_to_cam = np.linalg.inv(np.asarray(matrix_world, dtype=float))
        self.fx, self.fy = float(fx), abs(float(fy))
        self.cx, self.cy = float(cx), float(cy)  # principal point in units of image width and height
        self.width, self.height = (float(v) for v in image_size)

    @classmethod
    def from_dict(cls, camera: dict, image_size) -> "Camera":
        """From a sample's `camera` record (`matrix_world`, `fx`, `fy`, optional `cx`, `cy`)."""
        return cls(camera["matrix_world"], camera["fx"], camera["fy"], image_size, camera.get("cx", 0.5), camera.get("cy", 0.5))

    def project(self, points: np.ndarray) -> np.ndarray:
        """(..., 3) world points -> (..., 2) pixels; NaN where a point is behind the camera."""
        cam = points @ self.world_to_cam[:3, :3].T + self.world_to_cam[:3, 3]
        depth = -cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (self.cx + self.fx * cam[..., 0] / depth) * self.width
            v = (self.cy - self.fy * cam[..., 1] / depth) * self.height
        uv = np.stack([u, v], axis=-1)
        uv[depth <= 1e-9] = np.nan
        return uv

    def project_box(self, points: np.ndarray) -> np.ndarray:
        """Bounds of the projected convex outline visible inside the image.

        Clipping a bounding rectangle is insufficient: a vertex above the
        image can determine its left edge, while the visible silhouette's
        left edge is an intersection with the top border. Clip the projected
        outline itself. Entirely off-screen hypotheses retain their original
        bounds as a continuous optimisation extension; no observation exists
        there. Point/centre projections retain their original behaviour.
        """
        uv = self.project(points)
        lo, hi = uv.min(axis=-2), uv.max(axis=-2)
        bounds = np.concatenate([lo, hi], axis=-1)
        if uv.shape[-2] < 3:
            return bounds
        flat_uv = uv.reshape(-1, uv.shape[-2], 2)
        flat_bounds = bounds.reshape(-1, 4)
        cut = ((flat_bounds[:, 0] < 0) | (flat_bounds[:, 1] < 0)
               | (flat_bounds[:, 2] > self.width) | (flat_bounds[:, 3] > self.height))
        if not cut.any():
            return bounds
        import cv2

        for i in np.flatnonzero(cut):
            vertices = flat_uv[i]
            if not np.isfinite(vertices).all():
                continue
            # OpenCV supplies only hull indices. Keep the original float64
            # coordinates so finite-difference optimisation retains precision.
            indices = cv2.convexHull(vertices.astype(np.float32), returnPoints=False)
            if indices is None or len(indices) < 3:
                continue
            polygon = vertices[indices.ravel()]
            for axis, limit, lower in ((0, 0., True), (0, self.width, False),
                                       (1, 0., True), (1, self.height, False)):
                clipped = []
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                    a_in = a[axis] >= limit if lower else a[axis] <= limit
                    b_in = b[axis] >= limit if lower else b[axis] <= limit
                    if a_in:
                        clipped.append(a)
                    if a_in != b_in:
                        t = (limit - a[axis]) / (b[axis] - a[axis])
                        cross = a + t * (b - a)
                        cross[axis] = limit
                        clipped.append(cross)
                polygon = np.asarray(clipped, dtype=float).reshape(-1, 2)
                if not len(polygon):
                    break
            if len(polygon):
                flat_bounds[i] = np.concatenate([polygon.min(axis=0), polygon.max(axis=0)])
        return bounds


def lookat_camera(position, look_at, fov_deg: float, image_size, up=(0.0, 0.0, 1.0), principal_offset_px=(0.0, 0.0)) -> dict:
    """`matrix_world`, `fx`, `fy` of a camera at `position` looking at `look_at` with vertical field of view
    `fov_deg` (the Genesis convention, as in PhyEditing), and `cx`, `cy` when the principal point is offset."""
    pos = np.asarray(position, dtype=float)
    forward = np.asarray(look_at, dtype=float) - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = right, np.cross(right, forward), -forward, pos
    fy = 0.5 / np.tan(np.radians(fov_deg) / 2.0)
    out = {"matrix_world": matrix.tolist(), "fx": fy * image_size[1] / image_size[0], "fy": fy}
    if any(principal_offset_px):
        out.update(cx=0.5 + principal_offset_px[0] / image_size[0], cy=0.5 + principal_offset_px[1] / image_size[1])
    return out


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
