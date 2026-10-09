"""Same-time oracle-box control for the existing cross-flight pilot; diagnostic only."""
import copy
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from gravtrace.camera import Camera,quat_to_matrix
from gravtrace.fit import safe_fit
from scripts.phyediting_joint_trial import joint_fit
from scripts.phyediting_cross_window_trial import build_jobs


def project_sample(sample,trajectory,name,start):
    s=copy.deepcopy(sample)
    times=np.asarray(s['boxes']['times']);fps=s['fps']
    offset=s['motion']['t0'][0]
    frames=np.rint(start+(times-times[0]+offset)*fps).astype(int)
    with np.load(Path('data/compact')/(trajectory+'.npz')) as z:
        j=list(map(str,z['names'])).index(name);corners=np.asarray(s['object']['corners'])
        points=np.stack([z['position'][f,j]+corners@quat_to_matrix(z['quaternion_xyzw'][f,j]).T for f in frames])
        position_delta=float(np.linalg.norm(np.asarray(s['object']['position'])-z['position'][start,j]))
        rotation_delta=float(np.linalg.norm(quat_to_matrix(s['object']['quaternion'])-quat_to_matrix(z['quaternion_xyzw'][start,j])))
        omega_delta=float(np.linalg.norm(np.asarray(s['object']['angular_velocity'])-z['angular_velocity'][start,j]))
    uv=Camera.from_dict(s['camera'],s['image_size']).project(points)
    full=np.hstack([uv.min(axis=1),uv.max(axis=1)]);width,height=s['image_size']
    boxes=np.clip(full,0,[width,height,width,height]);s['boxes']['xyxy']=boxes.tolist()
    return s,{'reference_frames':frames.tolist(),'tiny_or_offscreen_frames':int(((boxes[:,2:]-boxes[:,:2]).min(axis=1)<6).sum()),'declared_position_delta_m':position_delta,'declared_rotation_matrix_delta':rotation_delta,'declared_omega_delta':omega_delta}


def evaluate(job):
    meta,observed,oracle=job
    return {**meta,'independent_observed':[safe_fit(s) for s in observed],
            'independent_oracle':[safe_fit(s) for s in oracle],'joint_oracle':joint_fit(oracle)}


def main():
    jobs,items,baseline=build_jobs();oracle_windows={r['id']:r for r in map(json.loads,open('runs/all_oracle.jsonl'))}
    diag=[]
    for meta,samples in jobs:
        it=items[meta['id']];projections=[];audits=[]
        for k,s in enumerate(samples):
            start=it['window']['start_frame'] if k==0 else oracle_windows[s['id']]['window']['start_frame']
            projection,audit=project_sample(s,it['trajectory'],it['object_name'],start)
            projections.append(projection);audits.append(audit)
        diag.append(({**meta,'projection_audit':audits},samples,projections))
    with ProcessPoolExecutor(max_workers=6) as pool:records=list(pool.map(evaluate,diag))
    for r in records:
        r['target']=items[r['id']]['gravity']
        for key in ['independent_observed','independent_oracle']:
            r[key+'_errors_pct']=[abs(p['gravity']/r['target']-1)*100 if p['status']=='ok' else None for p in r[key]]
        p=r['joint_oracle'];r['joint_oracle_error_pct']=abs(p['gravity']/r['target']-1)*100 if p['status']=='ok' else None
    result={'scope':'same 12 previously selected development pilot cases; oracle projections at identical times, independent velocities and no target gravity passed to fitter; root-cause control only','records':records}
    Path('docs/phyediting/cross_window_diagnosis15.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in records:print(r['cohort'],r['id'],r['independent_observed_errors_pct'],r['independent_oracle_errors_pct'],r['joint_oracle_error_pct'])

if __name__=='__main__':main()
