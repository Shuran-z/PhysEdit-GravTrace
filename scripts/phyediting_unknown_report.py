"""Report unknown-velocity common-frame controls, retaining all failures and groups."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np


def stats(rows):
    errors=[r['error_pct'] for r in rows if r['error_pct'] is not None]
    return {'attempted':len(rows),'ok':len(errors),'failed_or_missing':len(rows)-len(errors),
            'status_counts':dict(Counter(r['status'] for r in rows)),
            'mean_pct':float(np.mean(errors)) if errors else None,'max_pct':max(errors) if errors else None,
            'within10_all':sum(e<=10 for e in errors),'within20_all':sum(e<=20 for e in errors)}


def main():
    items={r['id']:r for r in map(json.loads,open('runs/benchmark_gravity_v1/items.jsonl'))}
    samples={r['id']:r for r in map(json.loads,open('runs/unknown_common.jsonl'))}
    files={'ours_centred_experimental':Path('runs/unknown_common_ours.jsonl')}
    files.update({p.stem:p for p in Path('runs/unknown_common_baselines').glob('*.jsonl')
                  if (p.stem.endswith('_anchored') or p.stem.endswith('_raw')) or p.stem in ['pixel2d_agnostic','lift_oracle']})
    methods={}
    for name,path in files.items():
        preds={r['id']:r for r in map(json.loads,path.open())}
        if name.startswith('ours'):
            assert set(preds)==set(samples), 'partial fits must not be reported as complete'
            assert all(preds[i]['frames']==len(s['boxes']['frames']) for i,s in samples.items())
        rows=[]
        for i,it in items.items():
            pred=preds.get(i,{})
            status=pred.get('status','missing') if i in samples else 'no_common_frames'
            error=abs(pred['gravity']/it['gravity']-1)*100 if status=='ok' else None
            rows.append({'id':i,'status':status,'error_pct':error})
        groups={}
        for key in ['event','gravity','camera_id','physics_group']:
            bins=defaultdict(list)
            for r in rows:bins[str(items[r['id']][key])].append(r)
            groups[key]={k:stats(v) for k,v in bins.items()}
        means=[v['mean_pct'] for v in groups['physics_group'].values() if v['mean_pct'] is not None]
        methods[name]={**stats(rows),'physics_group_mean_pct':float(np.mean(means)) if means else None,'groups':groups}
    oracle = {r['id']:r for r in map(json.loads,files['ours_centred_experimental'].open())}
    for name,path in files.items():
        paired = [r for r in map(json.loads,path.open()) if r['id'] in oracle and r.get('status')=='ok' and oracle[r['id']].get('status')=='ok']
        methods[name]['paired_success_n'] = len(paired)
        methods[name]['ours_mean_on_paired_pct'] = float(np.mean([abs(oracle[r['id']]['gravity']/items[r['id']]['gravity']-1)*100 for r in paired])) if paired else None
    out={'scope':'unknown-velocity 30fps inspected development benchmark; no reference velocity, no error-based exclusion',
         'coverage':{'input':len(items),'common_eligible':len(samples),'no_common_frames':len(items)-len(samples)},
         'protocol':'identical complete visible unclipped fit frames; depth missing any retained frame fails instead of silently shortening the fit; ours uses centred offsets and no contact truncation',
         'caveats':['2D/lifting use free linear velocity with no 8m/s norm bound; ours uses declared 0-8m/s bound. Same supplied declarations do not mean identical priors or representation.',
                    'Only ours models declared geometry and spin forward. Lift-oracle receives extra true depth.',
                    'Cached VGGT inference may include frames outside the retained fit subset; strict equality of neural input frames still needs a dedicated rerun before final claims.',
                    'Depth anchoring extrapolates model depth to declared start using a quadratic fit on retained frames, avoiding assigning the initial depth to a later frame. This is an experimental baseline scale estimate.',
                    '30fps common subset differs from the earlier 15fps tail experiment; not a new low-fps achievement or held-out test.'],
         'methods':methods}
    Path('docs/phyediting/unknown_common.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
    print(json.dumps({name:{k:v for k,v in m.items() if k!='groups'} for name,m in methods.items()},indent=2))

if __name__=='__main__':main()
