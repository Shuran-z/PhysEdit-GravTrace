"""Local gravity/velocity coupling at saved fits; no refit, oracle gravity or data rejection."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phyediting_gen import _read_tracks, _sample
from gravtrace.fit import _Problem


def conditional_information(jac):
    """Project the log-g column off nuisance columns; local linear diagnostic only."""
    gravity = jac[:, 0]
    nuisance = jac[:, 1:]
    remainder = gravity - nuisance @ np.linalg.lstsq(nuisance, gravity, rcond=1e-9)[0] if nuisance.size else gravity
    raw, conditional = np.linalg.norm(gravity), np.linalg.norm(remainder)
    return {'logg_column_norm_px': float(raw), 'conditional_logg_norm_px': float(conditional),
            'retained_information_fraction': float((conditional/raw)**2) if raw > 0 else 0.,
            'unit_iid_px_logg_scale': float(1/conditional) if conditional > 1e-12 else None,
            'jacobian_rank': int(np.linalg.matrix_rank(jac, tol=1e-8)), 'free_parameters': jac.shape[1]}


def diagnose(sample, prediction):
    n = prediction['frames']
    times = np.asarray(sample['boxes']['times'][:n]);times -= times[0]
    boxes = np.asarray(sample['boxes']['xyxy'][:n])
    visibility = sample['window'].get('edges_visible', {})
    seen = np.array([visibility.get(str(f), [True]*4) for f in sample['boxes']['frames'][:n]])
    problem = _Problem(sample, times, boxes, seen)
    h = problem.hypotheses()[prediction['hypothesis']]
    names = ['t0'] + h['names']
    x = np.array([np.log(prediction['gravity'])] + [prediction['params'][k] for k in names])
    bounds = [h['t0']] + h['bounds']
    free = [0] + [i+1 for i,b in enumerate(bounds) if b[1]-b[0] > 1e-12]
    columns = []
    for i in free:
        step = 1e-5 * max(abs(x[i]), 1.)
        plus,minus = x.copy(),x.copy();plus[i] += step;minus[i] -= step
        columns.append((problem.residual(h,plus)-problem.residual(h,minus))/(2*step))
    jac = np.column_stack(columns)
    return {**conditional_information(jac), 'used_frames':n,
            'usable_edge_observations':int((problem.weight>0).sum()),
            'local_residual_rms_px':float(np.sqrt(np.mean(problem.residual(h,x)**2)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fps', type=int, choices=(15,30), default=15)
    a = p.parse_args()
    items = {r['id']:r for r in map(json.loads,open('runs/benchmark_gravity_v1/items.jsonl'))}
    rows = {r['item_id']:r for r in map(json.loads,open(f'runs/genfloor_rows_{a.fps}.jsonl'))}
    source = f'runs/contact_off{a.fps}_v1.jsonl'
    saved = {r['id']:r['predictions']['centred'] for r in map(json.loads,open(source))}
    assert set(items)==set(rows)==set(saved), 'require complete paired control'
    tracks = _read_tracks(f'runs/genfloor_tracks_{a.fps}.jsonl')
    results = []
    for i,it in items.items():
        pred=saved[i];row=rows[i]
        rec={'id':i,'status':pred['status'],**{k:it[k] for k in ['event','camera_id','physics_group','gravity']}}
        if pred['status']=='ok':
            sample=_sample(row,it,tracks.get(row['sample_id'],{}),'agnostic',{})
            sample['offset_mode']='centred';sample['window']['contact_check']=False
            rec.update(diagnose(sample,pred))
            # Error is appended after the diagnostic; never influences the Jacobian.
            rec['error_pct']=abs(pred['gravity']/it['gravity']-1)*100
        results.append(rec)
    good=[r for r in results if r['status']=='ok']
    def reduce(rs):
        return {'n':len(rs), 'median_retained_fraction':float(np.median([r['retained_information_fraction'] for r in rs])),
                'median_conditional_norm_px':float(np.median([r['conditional_logg_norm_px'] for r in rs]))} if rs else {'n':0}
    groups = {}
    for key in ['event','camera_id','physics_group','gravity']:
        grouped = defaultdict(list)
        for r in results: grouped[str(r[key])].append(r)
        groups[key] = {}
        for k,rs in grouped.items():
            successful = [r for r in rs if r['status']=='ok']
            errors = [r['error_pct'] for r in successful]
            groups[key][k] = {**reduce(successful), 'attempted':len(rs),
                              'failed_or_bound':len(rs)-len(successful),
                              'mean_error_pct':float(np.mean(errors)) if errors else None,
                              'max_error_pct':max(errors) if errors else None}
    out={'scope':'full inspected development set; local unrobust weighted Jacobian at saved contact-off centred fits, no new predictions',
         'caveat':'unit iid pixel scale is a hypothetical linear sensitivity, not calibrated uncertainty; correlated segmentation error and boundary/nonlinear effects are not covered; no threshold or rejection applied',
         'fps':a.fps,'attempted':len(results),'status_counts':dict(Counter(r['status'] for r in results)),
         'groups':groups,'summary':reduce(good),'error_over10':reduce([r for r in good if r['error_pct']>10]),
         'error_under3':reduce([r for r in good if r['error_pct']<3]),'records':results}
    Path(f'docs/phyediting/identifiability{a.fps}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('records','groups')},indent=2))

if __name__=='__main__':main()
