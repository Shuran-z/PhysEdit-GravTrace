import numpy as np
from scripts.phyediting_joint_coverage import observed_frames
from scripts.phyediting_joint_trial import joint_fit
from tests.test_synthetic import sample, render, ballistic, FRAMES, FPS, X0


def test_auxiliary_subsampling_uses_actual_condition_grid():
    tr={'frames':[22,23,24,25,26,27], 'xyxy':[[f]*4 for f in range(22,28)]}
    assert [f for f,b in observed_frames(tr,24,27)]==[25,27]


def test_joint_recovers_shared_gravity_with_independent_velocities():
    samples=[]
    for velocity in [[1.,0.,.6],[-.4,.1,.3]]:
        s=sample('projectile',{'t0':[0.,0.],'speed':[0.,8.],'angle_deg':[-89.,89.]},8)
        s['offset_mode']='centred'
        s['boxes']={'frames':FRAMES[:8].tolist(),'times':(FRAMES[:8]/FPS).tolist(),
                    'xyxy':render(ballistic(FRAMES[:8]/FPS,X0,np.array(velocity),3.71,np.array([0.,0.,-1.]))).tolist()}
        s['truth']={'gravity':70.}
        samples.append(s)
    fit=joint_fit(samples)
    assert fit['status']=='ok' and abs(fit['gravity']/3.71-1)<.06
    assert fit['objects']==2
