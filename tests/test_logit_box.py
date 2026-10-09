import numpy as np
import pytest

from scripts.track_sam2 import logit_box


def rectangle(x0, y0, x1, y1, width=30, height=20):
    x, y = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
    return np.minimum.reduce([x - x0, x1 - x, y - y0, y1 - y])


def test_subpixel_recovers_continuous_edges_and_pixel_mode_is_unchanged():
    target = [5.2, 4.3, 23.4, 16.7]
    logits = rectangle(*target)
    assert np.allclose(logit_box(logits, "subpixel"), target)
    assert logit_box(logits) == [5, 4, 23, 17]
    shifted = [v + .15 for v in target]
    assert np.allclose(logit_box(rectangle(*shifted), "subpixel"), shifted)


def test_empty_clipped_and_nonfinite_regions():
    assert logit_box(-np.ones((4, 5)), "subpixel") is None
    assert logit_box(np.ones((4, 5)), "subpixel") == [0., 0., 5., 4.]
    assert logit_box(np.full((4, 5), np.nan)) is None
    with pytest.raises(ValueError):
        logit_box(np.ones(4))
