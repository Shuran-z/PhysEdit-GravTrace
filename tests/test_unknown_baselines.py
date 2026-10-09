import numpy as np
from scripts.baselines import methods_for,camera_frame,projection_jacobian
from scripts.phyediting_unknown_common import remove_velocity
from tests.test_synthetic import sample


def test_unknown_2d_recovers_declared_direction_without_reference_velocity():
    s=remove_velocity(sample('projectile',{'v0':[999,888,777],'gravity_dir':[0,0,-1],'directions':[[1,0,0]]},8))
    assert 'v0' not in s['motion'] and 'directions' not in s['motion']
    R,T,K=camera_frame(s);J=projection_jacobian(R@np.array(s['object']['position'])+T,K)@R
    frames=np.arange(8);t=frames/s['fps'];gpx=J@np.array([0,0,-1])
    uv=np.array([320.,180.])+t[:,None]*[17,-5]+.5*t[:,None]**2*3.71*gpx
    boxes=np.column_stack([uv-10,uv+10]);s['boxes']={'frames':frames.tolist(),'xyxy':boxes.tolist()};s['window']['start_frame']=0
    pred=methods_for(s,{'fake':{'frames':frames.tolist(),'depth':[4.]*7+[None],'relative':False}},None)
    assert abs(pred['pixel2d_agnostic']/3.71-1)<1e-9
    assert pred['lift_fake_anchored'] is None
    assert not any(k.endswith('_v0') for k in pred)


def test_depth_scale_uses_declared_start_extrapolation_for_delayed_frames(monkeypatch):
    import scripts.baselines as baselines
    monkeypatch.setattr(baselines,'surface_depth',lambda *args:4.)
    s=remove_velocity(sample('projectile',{'v0':[999,888,777],'gravity_dir':[0,0,-1]},8))
    frames=np.arange(2,10);t=frames/s['fps']
    uv=np.column_stack([320+3*t,180+5*t*t])
    s['boxes']={'frames':frames.tolist(),'xyxy':np.column_stack([uv-10,uv+10]).tolist()}
    s['window']['start_frame']=0
    depth=4+2*t+3*t*t
    pred=methods_for(s,{'fake':{'frames':frames.tolist(),'depth':depth.tolist(),'relative':False}},None)
    assert abs(pred['lift_fake_raw']-pred['lift_fake_anchored'])<1e-9
