"""Fixed common-frame auxiliary admission paired control; every primary item retained."""
import argparse
import copy
from collections import Counter,defaultdict
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scripts.phyediting_cross_window_trial import build_jobs
from scripts.phyediting_joint_trial import joint_fit
from gravtrace.fit import safe_fit


def common_auxiliary(sample):
    s=copy.deepcopy(sample);b=np.asarray(s['boxes']['xyxy']);frames=s['boxes']['frames'];times=np.asarray(s['boxes']['times'])
    w,h=s['image_size'];limits=np.array([w,h,w,h]);seen=s['window'].get('edges_visible',{})
    v=np.array([seen.get(str(f),[True]*4) for f in frames],bool)
    keep=np.flatnonzero(np.all(v&(b>1)&(b<limits-1),axis=1))
    if len(keep)<4:return None,len(keep)
    shift=float(times[keep[0]]-times[0]);s['motion']['t0']=[float(t)+shift for t in s['motion']['t0']]
    s['boxes']={'frames':[frames[k] for k in keep],'xyxy':b[keep].tolist(),'times':times[keep].tolist()}
    s['window']['max_frames']=len(keep)
    return s,len(keep)


def evaluate(job):
    meta,samples=job;primary,aux=samples;common,n=common_auxiliary(aux)
    pred=joint_fit([primary,common]) if common is not None and len(primary["boxes"]["frames"])>=4 else safe_fit(primary)
    return {**meta,'auxiliary_admitted':common is not None,'auxiliary_common_frames':n,'prediction':pred}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--all-items",action="store_true");args=parser.parse_args()
    jobs,items,baseline=build_jobs(all_eligible=args.all_items)
    with ProcessPoolExecutor(max_workers=6) as pool:records=list(pool.map(evaluate,jobs))
    if args.all_items:
        done={r['id'] for r in records}
        records.extend({'id':i,'cohort':'no_auxiliary','auxiliary_admitted':False,'auxiliary_common_frames':0,'prediction':baseline[i]} for i in items if i not in done)
    for r in records:r.update(target=items[r['id']]['gravity'],baseline=baseline[r['id']])
    summary={}
    for cohort in ['all','selected_tail','hash_control']:
        rs=[r for r in records if cohort=='all' or r['cohort']==cohort];summary[cohort]={}
        for method in ['baseline','prediction']:
            es=[abs(r[method]['gravity']/r['target']-1)*100 for r in rs if r[method]['status']=='ok']
            summary[cohort][method]={'attempted':len(rs),'ok':len(es),'failed_or_bound':len(rs)-len(es),'mean_pct':float(np.mean(es)) if es else None,'max_pct':max(es) if es else None}
    out={'scope':('all 1038 development items, fixed rule, not held-out' if args.all_items else 'same 4 error-selected development tails plus 8 hash controls; not overall or unseen validation'),
         'rule':'all declared auxiliary edges visible; observation >1px from borders; >=4 real frames; otherwise use original primary; all primary items/frames retained; no target-error fallback',
         'caveat':'extra simulator declarations and observations differ from single-window baselines; finite direction hypotheses; no global-optimality guarantee',
         'summary':summary,'records':records}
    if args.all_items:
        out['groups']={}
        for key in ['event','gravity','camera_id','physics_group']:
            bins=defaultdict(list)
            for r in records:bins[str(items[r['id']][key])].append(r)
            out['groups'][key]={}
            for k,rs in bins.items():
                es=[abs(r['prediction']['gravity']/r['target']-1)*100 for r in rs if r['prediction']['status']=='ok']
                out['groups'][key][k]={'attempted':len(rs),'ok':len(es),'status_counts':dict(Counter(r['prediction']['status'] for r in rs)),'mean_pct':float(np.mean(es)) if es else None,'max_pct':max(es) if es else None}
    stem='auxiliary_common_full15' if args.all_items else 'auxiliary_common_trial15'
    Path('docs/phyediting/'+stem+'.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
