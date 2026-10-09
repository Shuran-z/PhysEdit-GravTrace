"""Profile fixed gravity on development cases; optimize only declared nuisance ranges."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phyediting_gen import _read_tracks, _sample
from gravtrace.fit import _Problem, LOSS_SCALE_PX


def profile(problem, prediction, gravities):
    curves = []
    for gravity in gravities:
        best = None
        for h in problem.hypotheses():
            bounds = [h['t0']] + h['bounds']
            lo, hi = np.array(bounds).T
            free = np.flatnonzero(hi-lo > 1e-12)
            names = ['t0']+h['names']
            seeds = [(lo+hi)/2, np.clip([prediction['params'][k] for k in names], lo, hi)]
            for seed in seeds:
                def residual(z):
                    extra=seed.copy();extra[free]=z
                    return problem.residual(h,np.r_[np.log(gravity),extra])
                if len(free):
                    fit=least_squares(residual,seed[free],bounds=(lo[free],hi[free]),loss='soft_l1',f_scale=LOSS_SCALE_PX)
                    cost=float(fit.cost)
                else:
                    r=residual(np.array([]))/LOSS_SCALE_PX
                    cost=float(LOSS_SCALE_PX**2*np.sum(np.sqrt(1+r*r)-1))
                if best is None or cost<best:best=cost
        curves.append({'gravity':float(gravity),'cost':best})
    return curves


def evaluate(job):
    meta,sample,pred=job
    t=np.asarray(sample['boxes']['times']);t-=t[0]
    seen=sample['window']['edges_visible']
    problem=_Problem(sample,t,np.asarray(sample['boxes']['xyxy']),
                     np.array([seen.get(str(f),[True]*4) for f in sample['boxes']['frames']]))
    # Grid is centred on the existing observation-only estimate, never on target g.
    grid=np.clip(pred['gravity']*np.linspace(.5,1.8,27),.1,80)
    grid=np.unique(np.r_[grid,pred['gravity']])
    curve=profile(problem,pred,grid)
    winner=min(curve,key=lambda r:r['cost'])
    return {**meta,'saved_gravity':pred['gravity'],'saved_cost':pred['cost'],
            'minimum_on_grid':winner,'curve':curve}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    items={r['id']:r for r in map(json.loads,open('runs/benchmark_gravity_v1/items.jsonl'))}
    saved={r['id']:r['predictions']['centred'] for r in map(json.loads,open('runs/contact_off15_v1.jsonl'))}
    selected=[i for i,pred in saved.items() if pred['status']=='ok' and abs(pred['gravity']/items[i]['gravity']-1)>.1]
    controls=sorted([i for i,pred in saved.items() if pred['status']=='ok' and i not in selected],
                    key=lambda i:hashlib.sha256(('profile-v1:'+i).encode()).hexdigest())[:5]
    rows={r['item_id']:r for r in map(json.loads,open('runs/genfloor_rows_15.jsonl'))}
    tracks=_read_tracks('runs/genfloor_tracks_15.jsonl')
    jobs=[]
    for i in selected+controls:
        it,row=items[i],rows[i]
        sample=_sample(row,it,tracks.get(row['sample_id'],{}),'agnostic',{})
        sample['offset_mode']='centred';sample['window']['contact_check']=False
        # No truth is passed into the profile worker.
        jobs.append(({'id':i,'cohort':'error_selected_tail' if i in selected else 'hash_control'},sample,saved[i]))
    records=[]
    out=Path('runs/gravity_profile15_v1.jsonl')
    with ProcessPoolExecutor(max_workers=a.workers) as pool,out.open('w') as f:
        for r in pool.map(evaluate,jobs):
            target=items[r['id']]['gravity'];r['gravity_target']=target
            r['saved_error_pct']=abs(r['saved_gravity']/target-1)*100
            r['grid_error_pct']=abs(r['minimum_on_grid']['gravity']/target-1)*100
            r['relative_cost_improvement']=(r['saved_cost']-r['minimum_on_grid']['cost'])/max(r['saved_cost'],1e-12)
            f.write(json.dumps(r)+'\n');f.flush();records.append(r)
    result={'scope':'14 error-selected 15fps cases plus 5 fixed-hash controls, development diagnosis only, no global optimality guarantee',
            'protocol':'same full declared window, centred residuals, no contact trimming; fixed-g nuisance optimization, grid relative to saved estimate; target appended only after worker completion',
            'records':records}
    Path('docs/phyediting/gravity_profile15.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{'cohort':r['cohort'],'saved_error_pct':r['saved_error_pct'],'grid_error_pct':r['grid_error_pct'],
                      'cost_improvement':r['relative_cost_improvement']} for r in records],indent=2))

if __name__=='__main__':main()
