"""Declared-prior linear-motion control; not a claim that model representations match."""
import numpy as np
from scipy.optimize import minimize


def bounded_motion(t, observations, velocity_map, gravity_direction, speed_max=8., angle_max_deg=89.):
    """Fit offset, 3D velocity and g with the same speed/elevation/gravity bounds.

    Projection may be linearised (2D) or identity (lifted 3D). No velocity truth.
    SLSQP success and active gravity bounds are reported rather than hidden.
    """
    t=np.asarray(t,float);y=np.asarray(observations,float);M=np.asarray(velocity_map,float)
    direction=np.asarray(gravity_direction,float);direction/=np.linalg.norm(direction)
    dimensions=y.shape[1];A=np.zeros((y.size,dimensions+4))
    for k in range(dimensions):A[k::dimensions,k]=1.
    A[:,dimensions:dimensions+3]=(t[:,None,None]*M[None,:,:]).reshape(-1,3)
    A[:,-1]=(.5*t[:,None]**2*(M@direction)[None,:]).ravel()
    target=y.ravel();start=np.linalg.lstsq(A,target,rcond=None)[0]
    start[-1]=np.clip(start[-1],.1,80.)
    v=start[dimensions:dimensions+3];norm=np.linalg.norm(v)
    if norm>speed_max:v*=speed_max/norm
    sine2=np.sin(np.radians(angle_max_deg))**2
    def objective(x):
        residual=A@x-target
        return float(np.sum(np.sqrt(1+residual**2)-1))
    def gradient(x):
        r=A@x-target;return A.T@(r/np.sqrt(1+r*r))
    def speed(x):return speed_max**2-np.sum(x[dimensions:dimensions+3]**2)
    def elevation(x):
        v=x[dimensions:dimensions+3];return sine2*(v@v)-(v@direction)**2
    bounds=[(None,None)]*dimensions+[(-speed_max,speed_max)]*3+[(.1,80.)]
    seeds=[start.copy(),np.r_[np.mean(y,axis=0),np.zeros(3),4.]]
    fits=[minimize(objective,s,method='SLSQP',jac=gradient,bounds=bounds,
                   constraints=[{'type':'ineq','fun':speed},{'type':'ineq','fun':elevation}],
                   options={'maxiter':500,'ftol':1e-9}) for s in seeds]
    valid=[r for r in fits if r.success and speed(r.x)>=-1e-6 and elevation(r.x)>=-1e-6]
    if not valid:return {'status':'optimizer_failed'}
    best=min(valid,key=lambda r:r.fun);g=float(best.x[-1])
    return {'status':'ok' if .101<g<79.2 else 'at_bound','gravity':g,
            'velocity':best.x[dimensions:dimensions+3].tolist(),'cost':float(best.fun)}
