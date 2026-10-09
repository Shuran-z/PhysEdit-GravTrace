"""Gravity from a tracked target box.

The object's 8 bounding-box corners are placed on an analytic trajectory, projected with the
known camera, and the image box they span is compared with the observed mask box through the
displacement of each box edge since an anchor frame. A constant offset between declared geometry
and mask (pivot, mesh scale, mask bias) cancels, and edges cut by the image border drop out.

A robust least-squares fit returns g together with whatever the sample leaves undeclared, each
kept inside its declared range: `t0`, the time of the first observed frame after the declared
initial state, and the launch speed/elevation/azimuth (projectile) or along-slope speed and
distance to the top edge (incline). Discrete hypotheses (slope orientation) and azimuth seeds are
fitted separately and the lowest cost wins. Nothing here reads the sample's `truth`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .camera import Camera, quat_to_matrix
from .motion import ballistic, incline
from .observe import load_boxes

G_RANGE = (0.1, 80.0)
G_SEEDS = (1.0, 4.0, 16.0, 50.0)
DEFAULT_FRAMES = {"freefall": 6, "projectile": 8, "incline": 16}
MIN_FRAMES = 4
LOSS_SCALE_PX = 1.0  # soft-L1 knee: residuals beyond about a pixel count linearly
CONTACT_PX, CONTACT_SIGMA = 3.0, 3.0  # contact: a frame the preceding flight misses by > max(3 px, 3 x its median error)


def fit_sample(sample: dict) -> dict:
    masks = sample["masks"]
    frames, boxes = load_boxes(masks["dir"], masks["pattern"], masks.get("label"))
    return fit_boxes(sample, frames, boxes)


def fit_boxes(sample: dict, frames: np.ndarray, boxes: np.ndarray) -> dict:
    """Fit one sample from its observed boxes.

    Returns the sample id, `status` ("ok"; "at_bound": g at a search limit, e.g. an object that
    never falls; "few_frames"/"no_track": too few usable observations), `gravity` (m/s^2), the
    `frames` used, the fitted nuisances (`params`) and `sensitivity_px`, the image change between
    g/2 and 2g: a few pixels or less means this video barely constrains g.
    """
    out = {"id": sample["id"], "scenario": sample["scenario"]}
    window = sample.get("window", {})
    keep = frames >= window.get("start_frame", 0)  # frames before the declared motion start are not modelled
    frames, boxes = frames[keep], boxes[keep]
    max_frames = window.get("max_frames") or DEFAULT_FRAMES[sample["scenario"]]
    n = min(len(frames), int(max_frames))
    if n < MIN_FRAMES:
        return {**out, "status": "no_track", "frames": int(len(frames))}
    t = (frames - frames[0]) / float(sample["fps"])
    problem = _Problem(sample, t[:n], boxes[:n])
    if problem.anchor is None:
        return {**out, "status": "few_frames", "frames": n}
    best = problem.solve()
    contact = problem.contact_frame(best)
    if contact is not None and MIN_FRAMES <= contact < n:  # the analytic model ends at first contact
        n = contact
        problem = _Problem(sample, t[:n], boxes[:n])
        best = problem.solve() if problem.anchor is not None else best
    while n > MIN_FRAMES:  # contact with anything undeclared: the flight so far fails to predict the last frame
        shorter = _Problem(sample, t[:n - 1], boxes[:n - 1])
        if shorter.anchor is None:
            break
        short_best = shorter.solve()
        miss = problem.frame_error({"h": problem.hypotheses()[short_best["hypothesis"]], "x": short_best["x"]})[-1]
        if not _is_outlier(miss, shorter.frame_error(short_best)):
            break
        n, problem, best = n - 1, shorter, short_best
    g = float(np.exp(best["x"][0]))
    at_bound = not (G_RANGE[0] * 1.01 < g < G_RANGE[1] * 0.99)
    return {**out, "status": "at_bound" if at_bound else "ok", "gravity": g, "cost": best["cost"], "frames": n,
            "features": "box" if problem.use_size else "centre", "hypothesis": best["hypothesis"],
            "params": dict(zip(best["names"], map(float, best["x"][1:]))), "sensitivity_px": problem.sensitivity(best)}


def _is_outlier(miss: float, errors: np.ndarray) -> bool:
    """A predicted-frame miss beyond max(CONTACT_PX, CONTACT_SIGMA x the fitted frames' median error)."""
    fitted = errors[errors > 0]  # the anchor frame fits exactly by construction
    return miss > max(CONTACT_PX, CONTACT_SIGMA * (np.median(fitted) if fitted.size else 0.0))


class _Problem:
    def __init__(self, sample: dict, t: np.ndarray, boxes: np.ndarray) -> None:
        self.m, self.t, self.scenario = sample["motion"], t, sample["scenario"]
        obj = sample["object"]
        self.use_size = obj.get("corners") is not None  # without geometry only the box centre is modelled
        corners = np.asarray(obj["corners"], dtype=float) if self.use_size else np.zeros((1, 3))
        self.corners = corners @ quat_to_matrix(obj.get("quaternion")).T
        self.x0 = np.asarray(obj["position"], dtype=float)
        self.g_dir = np.asarray(self.m.get("gravity_dir", [0.0, 0.0, -1.0]), dtype=float)
        self.g_dir /= np.linalg.norm(self.g_dir)
        self.camera = Camera(sample["camera"]["matrix_world"], sample["camera"]["fx"], sample["camera"]["fy"],
                             sample["image_size"])
        size = boxes[:, 2:] - boxes[:, :2]
        area = size.prod(axis=1)
        clipped = np.column_stack([boxes[:, 0] <= 1, boxes[:, 1] <= 1,
                                   boxes[:, 2] >= self.camera.width - 1, boxes[:, 3] >= self.camera.height - 1])
        if not self.use_size:  # centre only: a clipped edge displaces the centre on that axis
            clipped = np.tile(clipped[:, :2] | clipped[:, 2:], 2)
            boxes = np.tile(0.5 * (boxes[:, :2] + boxes[:, 2:]), 2)
        tiny = (size < 6).any(axis=1) | (area < 36)
        median = np.median(area[~tiny]) if (~tiny).any() else 1.0
        reliable = np.flatnonzero(~tiny & (area >= 0.35 * median))
        self.anchor = int(reliable[0]) if reliable.size else None
        # per-edge weights: small boxes count less; an edge on the image border (here or in the anchor) not at all
        frame_weight = np.where(tiny, 0.0, np.sqrt(np.minimum(area / median, 1.0)))
        self.weight = frame_weight[:, None] * ~clipped
        if self.anchor is not None:
            self.weight *= ~clipped[self.anchor]
        if (self.weight.sum(axis=1) > 0).sum() < 3:
            self.anchor = None
        self.obs = boxes

    # --- model ----------------------------------------------------------------------------
    def hypotheses(self) -> list[dict]:
        """Each hypothesis: parameter names, bounds, and a centre trajectory c(tau, g, extras)."""
        m, t0 = self.m, [float(v) for v in self.m.get("t0", [0.0, 0.0])]
        up = -self.g_dir
        if self.scenario == "incline":  # an upslope start may leave over the top edge, 0..length metres away
            speed = m.get("speed", [0.0, 0.0])
            hyps = []
            for s in m["slopes"]:
                if speed[0] < 0.0 and s.get("length"):
                    hyps.append(dict(names=["speed", "top"], bounds=[speed, [0.0, s["length"]]], t0=t0,
                                     centres=lambda tau, g, e, s=s: incline(tau, self.x0, e[0], g, s, self.g_dir, e[1])))
                else:
                    hyps.append(dict(names=["speed"], bounds=[speed], t0=t0,
                                     centres=lambda tau, g, e, s=s: incline(tau, self.x0, e[0], g, s, self.g_dir)))
            return hyps
        if "v0" in m or self.scenario == "freefall":
            v0 = np.asarray(m.get("v0", [0.0, 0.0, 0.0]), dtype=float)
            return [dict(names=[], bounds=[], t0=t0, centres=lambda tau, g, e: ballistic(tau, self.x0, v0, g, self.g_dir))]

        # unknown launch: speed and elevation within the declared ranges, azimuth free (seeded by any declared directions)
        e1 = np.eye(3)[0] if abs(up[0]) < 0.9 else np.eye(3)[1]
        e1 = (e1 - e1 @ up * up) / np.linalg.norm(e1 - e1 @ up * up)
        e2 = np.cross(up, e1)

        def launch(tau, g, e):
            d = np.cos(e[2]) * e1 + np.sin(e[2]) * e2
            return ballistic(tau, self.x0, e[0] * (np.cos(e[1]) * d + np.sin(e[1]) * up), g, self.g_dir)
        seeds = [np.arctan2(np.dot(d, e2), np.dot(d, e1)) for d in m.get("directions", [])]
        angle = np.radians(m["angle_deg"])
        return [dict(names=["speed", "angle", "azimuth"], bounds=[m["speed"], angle, [phi - np.pi, phi + np.pi]], t0=t0,
                     centres=launch) for phi in seeds or np.linspace(-np.pi, np.pi, 8, endpoint=False)]

    def predict(self, h: dict, x: np.ndarray) -> np.ndarray:
        """Image box [x0, y0, x1, y1] spanned by the projected corners in each frame."""
        g, t0, extras = np.exp(x[0]), x[1], x[2:]
        centres = h["centres"](np.maximum(self.t + t0, 0.0), g, extras)  # held in the initial state until release
        uv = self.camera.project(centres[:, None, :] + self.corners[None])
        return np.hstack([uv.min(axis=1), uv.max(axis=1)])

    def residual(self, h: dict, x: np.ndarray) -> np.ndarray:
        """Edge displacements since the anchor frame, predicted minus observed (px, weighted)."""
        pred = self.predict(h, x)
        if not np.isfinite(pred).all():
            return np.full(self.obs.size, 1e3)
        diff = (pred - pred[self.anchor]) - (self.obs - self.obs[self.anchor])
        return (diff * self.weight).ravel()

    # --- fitting --------------------------------------------------------------------------
    def solve(self) -> dict:
        best = None
        for index, h in enumerate(self.hypotheses()):
            lo = [np.log(G_RANGE[0]), h["t0"][0]] + [b[0] for b in h["bounds"]]
            hi = [np.log(G_RANGE[1]), h["t0"][1]] + [b[1] for b in h["bounds"]]
            fixed = [abs(b - a) < 1e-12 for a, b in zip(lo, hi)]
            free = [i for i, f in enumerate(fixed) if not f]
            for g0 in G_SEEDS:
                x = np.array([np.log(g0)] + [0.5 * (a + b) for a, b in zip(lo[1:], hi[1:])])

                def fun(z, x=x):
                    x = x.copy()
                    x[free] = z
                    return self.residual(h, x)
                r = least_squares(fun, x[free], bounds=([lo[i] for i in free], [hi[i] for i in free]),
                                  loss="soft_l1", f_scale=LOSS_SCALE_PX)
                x[free] = r.x
                if best is None or r.cost < best["cost"]:
                    best = dict(cost=float(r.cost), x=x, h=h, hypothesis=index, names=["t0"] + h["names"])
        return best

    def contact_frame(self, best: dict) -> int | None:
        """First frame whose predicted lowest corner is below the declared floor (landing, or the ramp's foot)."""
        ground = self.m.get("ground_z")
        if ground is None:
            return None
        g, t0 = np.exp(best["x"][0]), best["x"][1]
        centres = best["h"]["centres"](np.maximum(self.t + t0, 0.0), g, best["x"][2:])
        z = (centres[:, None, :] + self.corners[None])[..., 2].min(axis=1)
        below = np.flatnonzero((z < ground) & (np.arange(len(z)) > 0))
        return int(below[0]) if below.size and z[0] >= ground else None

    def frame_error(self, best: dict) -> np.ndarray:
        """Per-frame RMS edge residual (px) of a fit."""
        r = self.residual(best["h"], best["x"]).reshape(-1, 4)
        return np.sqrt((r ** 2).sum(axis=1) / np.maximum((self.weight > 0).sum(axis=1), 1))

    def sensitivity(self, best: dict) -> float:
        """Feature change (px RMS) between g/2 and 2g at the fitted nuisances: how well g is observable."""
        lo, hi = best["x"].copy(), best["x"].copy()
        lo[0] -= np.log(2.0)
        hi[0] += np.log(2.0)
        return float(np.sqrt(np.mean((self.residual(best["h"], hi) - self.residual(best["h"], lo)) ** 2)))
