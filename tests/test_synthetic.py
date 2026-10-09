"""End-to-end recovery on synthetic tracks: render boxes from a known motion, fit, compare."""
from pathlib import Path

import numpy as np
import pytest

import gravtrace
from gravtrace.camera import Camera
from gravtrace.fit import fit_boxes
from gravtrace.motion import ballistic, incline

FPS = 16.0
IMAGE = (640, 480)
CAMERA = {"matrix_world": [[1, 0, 0, 0], [0, 0, -1, -4], [0, 1, 0, 1], [0, 0, 0, 1]], "fx": 1.0, "fy": 640 / 480}
CORNERS = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05)])
X0 = np.array([0.0, 0.0, 1.2])
FRAMES = np.arange(12)


def render(centres: np.ndarray) -> np.ndarray:
    """Pixel-coverage box of the projected corners, quantised like a mask."""
    uv = Camera(CAMERA["matrix_world"], CAMERA["fx"], CAMERA["fy"], IMAGE).project(centres[:, None, :] + CORNERS[None])
    return np.hstack([np.floor(uv.min(axis=1)), np.ceil(uv.max(axis=1))])


def sample(scenario: str, motion: dict, max_frames: int) -> dict:
    return {"id": "synthetic", "scenario": scenario, "fps": FPS, "image_size": IMAGE, "camera": CAMERA,
            "object": {"corners": CORNERS.tolist(), "position": X0.tolist(), "quaternion": [0, 0, 0, 1]},
            "motion": motion, "window": {"max_frames": max_frames}}


@pytest.mark.parametrize("g", [1.62, 9.81, 15.0])
def test_freefall(g):
    t0 = 1.0 / FPS
    boxes = render(ballistic(FRAMES / FPS + t0, X0, np.zeros(3), g, np.array([0.0, 0.0, -1.0])))
    fit = fit_boxes(sample("freefall", {"t0": [t0, t0]}, 6), FRAMES, boxes)
    assert fit["status"] == "ok" and abs(fit["gravity"] / g - 1) < 0.05


@pytest.mark.parametrize("g", [3.71, 9.81])
def test_projectile_with_unknown_launch(g):
    t0, angle = 1.0 / FPS, np.radians(40.0)
    v0 = 2.5 * np.array([np.cos(angle), 0.0, np.sin(angle)])
    boxes = render(ballistic(FRAMES / FPS + t0, X0, v0, g, np.array([0.0, 0.0, -1.0])))
    motion = {"t0": [t0 - 0.01, t0 + 0.01], "speed": [1.0, 5.0], "angle_deg": [10.0, 80.0],
              "directions": [[1, 0, 0], [-1, 0, 0], [0, 1, 0]]}
    fit = fit_boxes(sample("projectile", motion, 8), FRAMES, boxes)
    azimuth = np.angle(np.exp(1j * fit["params"]["azimuth"]))
    assert fit["status"] == "ok" and abs(azimuth) < 0.1 and abs(fit["gravity"] / g - 1) < 0.08


def test_incline_starting_upslope():
    g, angle = 9.81, 25.0
    slope = {"dir": [np.cos(np.radians(angle)), 0.0, -np.sin(np.radians(angle))], "angle_deg": angle, "mu": 0.05}
    mirrored = {**slope, "dir": [-slope["dir"][0], 0.0, slope["dir"][2]]}
    boxes = render(incline(FRAMES / FPS, X0, -1.5, g, slope, [0.0, 0.0, -1.0]))
    motion = {"t0": [0.0, 0.0], "speed": [-3.0, 0.0], "slopes": [slope, mirrored]}
    fit = fit_boxes(sample("incline", motion, 12), FRAMES, boxes)
    assert fit["status"] == "ok" and fit["hypothesis"] == 0 and abs(fit["gravity"] / g - 1) < 0.05


def test_incline_launch_over_the_top_edge():
    g, angle = 9.81, 25.0
    slope = {"dir": [np.cos(np.radians(angle)), 0.0, -np.sin(np.radians(angle))], "angle_deg": angle, "mu": 0.05, "length": 1.0}
    boxes = render(incline(FRAMES / FPS, X0, -3.0, g, slope, [0.0, 0.0, -1.0], top=0.4))
    fit = fit_boxes(sample("incline", {"t0": [0.0, 0.0], "speed": [-4.0, 0.0], "slopes": [slope]}, 12), FRAMES, boxes)
    assert fit["status"] == "ok" and abs(fit["gravity"] / g - 1) < 0.05 and abs(fit["params"]["top"] - 0.4) < 0.05


def test_clip_starting_before_release():
    g, lead = 9.81, 0.25  # the object is held for the first 0.25 s of the clip
    boxes = render(ballistic(np.maximum(FRAMES / FPS - lead, 0.0), X0, np.zeros(3), g, np.array([0.0, 0.0, -1.0])))
    fit = fit_boxes(sample("freefall", {"t0": [-lead, -lead]}, 12), FRAMES, boxes)
    assert fit["status"] == "ok" and abs(fit["gravity"] / g - 1) < 0.05


def test_static_object_hits_lower_bound():
    boxes = render(np.repeat(X0[None], len(FRAMES), axis=0))
    fit = fit_boxes(sample("freefall", {"t0": [0.0, 0.0]}, 6), FRAMES, boxes)
    assert fit["status"] == "at_bound" and fit["gravity"] < 0.2


def test_fit_path_never_reads_truth():
    root = Path(gravtrace.__file__).parent
    for name in ("fit.py", "motion.py", "observe.py", "camera.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert '"truth"' not in text and "'truth'" not in text


def test_spinning_flat_box_with_declared_damping():
    """A thin tumbling plate changes its image box a lot; the declared spin, inertia and damping carry it exactly."""
    from gravtrace.camera import quat_to_matrix
    from gravtrace.motion import torque_free
    g, mass, damping, drag = 1.62, 0.38, 0.018, 0.002
    plate = np.array([[x, y, z] for x in (-0.105, 0.105) for y in (-0.074, 0.074) for z in (-0.007, 0.007)])
    extent = plate.max(axis=0) - plate.min(axis=0)
    inertia = mass / 12 * np.array([extent[1] ** 2 + extent[2] ** 2, extent[0] ** 2 + extent[2] ** 2, extent[0] ** 2 + extent[1] ** 2])
    v0, omega = np.array([0.6, 0.0, 0.4]), np.array([0.5, 3.0, 0.2])
    tau = FRAMES / FPS
    rot = torque_free(tau, quat_to_matrix(None), omega, inertia, damping=damping)
    pts = ballistic(tau, X0, v0, g, [0.0, 0.0, -1.0], drag / mass)[:, None, :] + np.einsum("kij,cj->kci", rot, plate)
    uv = Camera(CAMERA["matrix_world"], CAMERA["fx"], CAMERA["fy"], IMAGE).project(pts)
    boxes = np.hstack([uv.min(axis=1), uv.max(axis=1)])
    s = sample("projectile", {"t0": [0.0, 0.0], "v0": v0.tolist()}, 10)
    s["object"].update(corners=plate.tolist(), angular_velocity=omega.tolist(), mass=mass, inertia=inertia.tolist(),
                       angular_damping=damping, linear_damping=drag)
    fit = fit_boxes(s, FRAMES, boxes)
    error = abs(fit["gravity"] / g - 1)
    assert fit["status"] == "ok" and error < 0.002
    s["object"].pop("angular_velocity")  # the same track read as a non-rotating box is clearly worse
    assert abs(fit_boxes(s, FRAMES, boxes)["gravity"] / g - 1) > 5 * error


def test_lookat_camera_projects_the_target_to_the_image_centre():
    from gravtrace.camera import lookat_camera
    cam = lookat_camera([3.55, 0.95, 1.56], [3.55, 2.25, 1.16], 44.0, (1280, 720))
    uv = Camera(cam["matrix_world"], cam["fx"], cam["fy"], (1280, 720)).project(np.array([[3.55, 2.25, 1.16]]))
    assert np.allclose(uv, [[640, 360]], atol=1e-6)
