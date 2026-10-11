"""Visible silhouette regression tests for objects crossing image borders."""
import numpy as np

from gravtrace.camera import Camera


def camera():
    return Camera(np.eye(4), 1., 1., [100, 100], cx=0., cy=0.)


def world(uv):
    uv = np.asarray(uv, dtype=float)
    return np.column_stack([uv[:, 0] / 100, -uv[:, 1] / 100, -np.ones(len(uv))])


def test_top_clipping_changes_the_visible_left_edge():
    # The off-screen left vertex must not set the visible left boundary.
    points = world([[-30, -40], [50, -40], [50, 40], [10, 40]])
    np.testing.assert_allclose(camera().project_box(points), [0, 0, 50, 40])
    points = world([[-10, -40], [50, -40], [50, 40], [30, 40]])
    np.testing.assert_allclose(camera().project_box(points), [10, 0, 50, 40])


def test_outline_can_cover_image_without_a_vertex_inside_it():
    points = world([[-20, -20], [120, -20], [120, 120], [-20, 120]])
    np.testing.assert_allclose(camera().project_box(points), [0, 0, 100, 100])


def test_in_view_and_centre_projection_keep_exact_original_values():
    c = camera()
    points = world([[10, 15], [70, 20], [35, 80]])
    uv = c.project(points)
    np.testing.assert_array_equal(c.project_box(points), np.r_[uv.min(0), uv.max(0)])
    np.testing.assert_array_equal(c.project_box(points[:1]), np.r_[uv[0], uv[0]])


def test_clipped_projection_preserves_small_parameter_displacements():
    points = world([[-10, -40], [50, -40], [50, 40], [30, 40]])
    moved = points.copy()
    moved[:, 0] += 1e-9
    delta = camera().project_box(moved) - camera().project_box(points)
    np.testing.assert_allclose(delta[[0, 2]], [1e-7, 1e-7], atol=1e-12)
