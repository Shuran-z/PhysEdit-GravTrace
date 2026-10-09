import numpy as np
from scripts.phyediting_edge_diagnosis import edge_stats


def test_curvature_diagnostic_removes_offset_and_velocity_trend():
    t = np.linspace(0, 1, 12)
    u = t - t.mean()
    r = edge_stats(t, 100 + 30*u + 4*u**2)
    assert abs(r['curvature_coefficient_px'] - 4) < 1e-9
    linear = edge_stats(t, 100 + 30*u)
    assert linear['curvature_excursion_px'] < 1e-10
    assert edge_stats(t[:3], u[:3]) == {'frames': 3}
