"""Report complete auxiliary-eligible fits plus a fixed single-object fallback; no error-based routing."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def stats(rows):
    es=[r['error_pct'] for r in rows if r['error_pct'] is not None]
    return {'attempted':len(rows),'ok':len(es),'failed_or_bound':len(rows)-len(es),
            'mean_pct':float(np.mean(es)) if es else None,'max_pct':max(es) if es else None,
            'within10_all':sum(e<=10 for e in es),'within20_all':sum(e<=20 for e in es)}


def main():
    full=json.load(open('docs/phyediting/joint_full15.json'))
    coverage=json.load(open('docs/phyediting/joint_coverage15.json'))
    fits={r['id']:r['prediction'] for r in full['records']}
    expected={r['id'] for r in coverage['records'] if r['concurrent_auxiliary_windows']}
    assert set(fits)==expected and len(full['records'])==len(expected)
    base={r['id']:r['predictions']['centred'] for r in map(json.loads,open('runs/contact_off15_v1.jsonl'))}
    methods={}
    for mode in ['baseline','joint_if_auxiliary_else_single']:
        rows=[]
        for r in coverage['records']:
            pred=fits.get(r['id'],base[r['id']]) if mode!='baseline' else base[r['id']]
            rows.append({**r,'status':pred['status'], 'error_pct':abs(pred['gravity']/r['gravity']-1)*100 if pred['status']=='ok' else None})
        groups={}
        for key in ['event','camera_id','gravity','physics_group']:
            bins=defaultdict(list)
            for r in rows:bins[str(r[key])].append(r)
            groups[key]={k:stats(v) for k,v in bins.items()}
        methods[mode]={**stats(rows),'groups':groups}
    out={'scope':'complete inspected benchmark with predetermined auxiliary-availability routing; no selection by target error',
         'caveat':'joint uses additional objects; earlier single-object depth and 2D baselines are not matched to that information',
         'auxiliary_eligible':len(expected),'methods':methods}
    Path('docs/phyediting/joint_comparison15.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:{key:v for key,v in m.items() if key!='groups'} for k,m in methods.items()},indent=2))

if __name__=='__main__':main()
