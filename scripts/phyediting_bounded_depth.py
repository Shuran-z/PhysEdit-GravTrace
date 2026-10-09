"""Same-frame lifted-depth control with declared velocity bounds, no reference velocity."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from scripts.baselines import camera_frame,surface_depth
from scripts.phyediting_bounded_baseline import bounded_motion

MODELS=['depthpro','dav2_metric','zoedepth','moge2','unidepth_v2','vggt']


def lift_job(sample,record):
    frames=np.asarray(sample['boxes']['frames']);t=(frames-sample['window']['start_frame'])/sample['fps']
    by_frame=dict(zip(record['frames'],record['depth'])) if record else {}
    depth=np.array([by_frame.get(int(f),np.nan) for f in frames],float)
    if not (np.isfinite(depth)&(depth>0)).all():return None,'missing_depth'
    A=np.column_stack([np.ones_like(t),t,.5*t*t]);initial=float(np.linalg.lstsq(A,depth,rcond=None)[0][0])
    if initial<=0:return None,'invalid_scale'
    R,T,K=camera_frame(sample);depth=depth*surface_depth(sample,R,T)/initial
    boxes=np.asarray(sample['boxes']['xyxy']);uv=.5*(boxes[:,:2]+boxes[:,2:])
    rays=np.column_stack([(uv[:,0]-K[0,2])/K[0,0],(uv[:,1]-K[1,2])/K[1,1],np.ones(len(t))])
    world=(rays*depth[:,None]-T)@R
    return (t,world,np.asarray(sample['motion']['gravity_dir'])),None


def evaluate(job):
    meta,data=job
    if isinstance(data,str):return {**meta,'status':data}
    t,points,direction=data
    return {**meta,**bounded_motion(t,points,np.eye(3),direction)}


def summarize(records,items):
    methods={}
    for model in MODELS:
        preds={r['id']:r for r in records if r['model']==model}
        rows=[]
        for i,it in items.items():
            p=preds.get(i,{'status':'no_common_frames'})
            rows.append({'id':i,'status':p['status'],'error_pct':abs(p['gravity']/it['gravity']-1)*100 if p['status']=='ok' else None})
        def stats(rs):
            errors=[r['error_pct'] for r in rs if r['error_pct'] is not None]
            return {'attempted':len(rs),'ok':len(errors),'failed_or_bound':len(rs)-len(errors),
                    'status_counts':dict(Counter(r['status'] for r in rs)),
                    'mean_pct':float(np.mean(errors)) if errors else None,'max_pct':max(errors) if errors else None}
        groups={}
        for key in ['event','gravity','camera_id','physics_group']:
            bins=defaultdict(list)
            for r in rows:bins[str(items[r['id']][key])].append(r)
            groups[key]={k:stats(v) for k,v in bins.items()}
        methods[model]={**stats(rows),'groups':groups}
    return methods


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    samples=list(map(json.loads,open('runs/unknown_common.jsonl')))
    items={r['id']:r for r in map(json.loads,open('runs/benchmark_gravity_v1/items.jsonl'))}
    jobs=[]
    for model in MODELS:
        depths={r['id']:r for r in map(json.loads,open(f'runs/final_depth_{model}.jsonl'))}
        for s in samples:
            data,error=lift_job(s,depths.get(s['id']))
            jobs.append(({'id':s['id'],'model':model},error if error else data))
    records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,open('runs/bounded_depth.jsonl','w') as f:
        for r in pool.map(evaluate,jobs):
            records.append(r);f.write(json.dumps(r)+'\n');f.flush()
    expected={(s['id'],model) for s in samples for model in MODELS}
    assert len(records)==len(expected) and {(r['id'],r['model']) for r in records}==expected
    out={'scope':'bounded lifted-depth common-frame development control, no reference velocity',
         'protocol':'all retained depths required; model depth extrapolated to declared initial surface; velocity0-8m/s, elevation+-89deg, gravity0.1-80',
         'caveat':'metric residual soft-L1 scale is 1 metre, unlike pixel loss; geometry/spin/drag representation remains different. Cached VGGT neural input frames still require alignment; finite SLSQP starts do not prove global optimality.',
         'methods':summarize(records,items)}
    Path('docs/phyediting/bounded_depth.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
    print(json.dumps({k:{x:v for x,v in m.items() if x!='groups'} for k,m in out['methods'].items()},indent=2))

if __name__=='__main__':main()
