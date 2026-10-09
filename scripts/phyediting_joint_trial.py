"""Experimental two-object gravity fit; independent velocities, no target gravity input."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gravtrace.fit import _Problem, G_RANGE, G_SEEDS, LOSS_SCALE_PX
from scripts.phyediting_gen import _read_tracks, _sample


def joint_fit(samples):
    problems=[];solutions=[]
    for sample in samples:
        boxes=sample['boxes'];n=min(len(boxes['frames']),sample['window']['max_frames'])
        if n<4:return {'status':'few_frames'}
        times=np.asarray(boxes['times'][:n]);times-=times[0]
        seen=sample['window'].get('edges_visible',{})
        problem=_Problem(sample,times,np.asarray(boxes['xyxy'][:n]),
                         np.array([seen.get(str(f),[True]*4) for f in boxes['frames'][:n]]))
        if problem.anchor is None:return {'status':'few_frames'}
        problems.append(problem);solutions.append(problem.solve())
    lower=[np.log(G_RANGE[0])];upper=[np.log(G_RANGE[1])];initial=[solutions[0]['x'][0]];slices=[]
    for solution in solutions:
        h=solution['h'];bounds=[h['t0']]+h['bounds'];start=len(initial)
        initial.extend(solution['x'][1:]);lower.extend(b[0] for b in bounds);upper.extend(b[1] for b in bounds)
        slices.append(slice(start,len(initial)))
    lower,upper,initial=map(np.asarray,(lower,upper,initial));free=np.flatnonzero(upper-lower>1e-12)
    def residual(x):
        return np.concatenate([p.residual(s['h'],np.r_[x[0],x[sl]]) for p,s,sl in zip(problems,solutions,slices)])
    best=None
    for gravity in G_SEEDS+(float(np.exp(initial[0])),):
        seed=initial.copy();seed[0]=np.log(gravity)
        def fun(z):
            x=seed.copy();x[free]=z;return residual(x)
        fit=least_squares(fun,seed[free],bounds=(lower[free],upper[free]),loss='soft_l1',f_scale=LOSS_SCALE_PX)
        seed[free]=fit.x
        if best is None or fit.cost<best[0]:best=(fit.cost,seed)
    gravity=float(np.exp(best[1][0]))
    return {'status':'ok' if G_RANGE[0]*1.01<gravity<G_RANGE[1]*.99 else 'at_bound','gravity':gravity,
            'cost':float(best[0]),'objects':len(samples),'frame_counts':[len(p.t) for p in problems],
            'independent_gravities':[float(np.exp(s['x'][0])) for s in solutions]}


def evaluate(job):
    meta,primary,auxiliary=job
    return {**meta,'prediction':joint_fit([primary,auxiliary])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--limit',type=int,default=24);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--out',default='joint_pilot15',help='distinct artifact stem for pilot/full runs')
    a=p.parse_args()
    aux=list(map(json.loads,open('runs/joint_auxiliary15.jsonl')))
    aux.sort(key=lambda r:hashlib.sha256(('joint-pilot-v1:'+r['item_id']).encode()).hexdigest())
    if a.limit:aux=aux[:a.limit]
    items={r['id']:r for r in map(json.loads,open('runs/benchmark_gravity_v1/items.jsonl'))}
    rows={r['item_id']:r for r in map(json.loads,open('runs/genfloor_rows_15.jsonl'))}
    tracks=_read_tracks('runs/genfloor_tracks_15.jsonl')
    baseline={r['id']:r['predictions']['centred'] for r in map(json.loads,open('runs/contact_off15_v1.jsonl'))}
    jobs=[]
    for rec in aux:
        i=rec['item_id'];it,row=items[i],rows[i]
        sample=_sample(row,it,tracks.get(row['sample_id'],{}),'agnostic',{})
        sample['offset_mode']='centred';sample['window']['contact_check']=False
        jobs.append(({'id':i},sample,rec['auxiliary']))
    records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,open(f'runs/{a.out}_v1.jsonl','w') as f:
        for rec in pool.map(evaluate,jobs):
            it=items[rec['id']];rec['gravity_target']=it['gravity'];rec['baseline']=baseline[rec['id']]
            records.append(rec);f.write(json.dumps(rec)+'\n');f.flush()
    summary={}
    for key in ['baseline','prediction']:
        es=[abs(r[key]['gravity']/r['gravity_target']-1)*100 for r in records if r[key]['status']=='ok']
        summary[key]={'attempted':len(records),'ok':len(es),'failed_or_bound':len(records)-len(es),
                      'mean_pct':float(np.mean(es)) if es else None,'max_pct':max(es) if es else None}
    out={'scope':('fixed-hash pilot' if a.limit else 'all auxiliary-eligible cases')+' among development videos; not whole-benchmark or held out',
         'caveat':'extra object observations and simulator-declared windows/geometry; not information-matched to earlier single-object baselines; joint hypotheses seeded by independent fits',
         'summary':summary,'records':records}
    Path(f'docs/phyediting/{a.out}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
