import numpy as np
from scripts.phyediting_bounded_baseline import bounded_motion


def test_bounded_motion_recovers_unknown_velocity_and_respects_declared_speed():
    t=np.linspace(0,.8,14);v=np.array([1.,.2,.3]);d=np.array([0.,0.,-1.])
    y=np.array([2.,3.,4.])+t[:,None]*v+.5*t[:,None]**2*3.71*d
    r=bounded_motion(t,y,np.eye(3),d)
    assert r['status']=='ok' and abs(r['gravity']/3.71-1)<1e-4
    assert np.linalg.norm(r['velocity'])<=8.+1e-6
    r=bounded_motion(t,y,np.eye(3),d,speed_max=.5)
    assert r['status'] in ('ok','at_bound') and np.linalg.norm(r['velocity'])<=.5+1e-6
