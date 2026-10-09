import numpy as np
from scripts.phyediting_gravity_profile import profile


class QuadraticProblem:
    def hypotheses(self):
        return [{'t0':[0.,0.], 'bounds':[[0.,8.]],'names':['speed']}]
    def residual(self,h,x):
        g=np.exp(x[0]);v=x[2]
        return np.array([g+v-5,2*g+v-8])


def test_profile_reoptimizes_velocity_without_target_gravity_input():
    p={'params':{'t0':0.,'speed':4.}}
    curve=profile(QuadraticProblem(),p,[2.,3.,4.])
    assert min(curve,key=lambda r:r['cost'])['gravity']==3.
    assert curve[1]['cost']<1e-12
    assert curve[0]['cost']>0 and curve[2]['cost']>0
