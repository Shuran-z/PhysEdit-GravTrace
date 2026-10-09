import numpy as np
from scripts.phyediting_identifiability import conditional_information


def test_nuisance_projection_detects_confounded_gravity_and_is_scale_invariant():
    g = np.array([1.,2.,3.,4.])
    assert conditional_information(np.column_stack([g,g]))['conditional_logg_norm_px'] < 1e-12
    j = np.column_stack([g, np.ones(4)])
    a = conditional_information(j)
    b = conditional_information(j * [1,100])
    assert abs(a['retained_information_fraction'] - b['retained_information_fraction']) < 1e-12
    assert conditional_information(g[:,None])['retained_information_fraction'] == 1
