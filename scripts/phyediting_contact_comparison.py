"""Compare complete contact-on/off trials; retain every item and report actual frame changes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np


def summary(rows):
    errors = [r['error_pct'] for r in rows if r['error_pct'] is not None]
    return {'attempted': len(rows), 'ok': len(errors), 'failed_or_bound': len(rows)-len(errors),
            'status_counts': dict(Counter(r['status'] for r in rows)),
            'coverage': len(errors)/len(rows) if rows else None,
            'mean_pct': float(np.mean(errors)) if errors else None,
            'max_pct': max(errors) if errors else None,
            'within10_all': sum(e <= 10 for e in errors), 'within20_all': sum(e <= 20 for e in errors)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fps', type=int, choices=(15,30), default=15)
    a = p.parse_args()
    items = {r['id']: r for r in map(json.loads, open('runs/benchmark_gravity_v1/items.jsonl'))}
    old = {r['id']: r for r in map(json.loads, open(f'runs/offset_full{a.fps}_v1.jsonl'))}
    new = {r['id']: r for r in map(json.loads, open(f'runs/contact_off{a.fps}_v1.jsonl'))}
    assert set(old) == set(new) == set(items), 'partial runs must not be presented as whole-benchmark results'
    records, methods = [], {}
    for mode, source in [('contact_on',old),('contact_off',new)]:
        rows = []
        for i,r in source.items():
            pred = r['predictions']['centred']
            rows.append({'id':i, 'status':pred['status'], 'frames':pred.get('frames'),
                         'error_pct':abs(pred['gravity']/items[i]['gravity']-1)*100 if pred['status']=='ok' else None})
        groups = {}
        for key in ['event','gravity','camera_id','physics_group']:
            grouped = defaultdict(list)
            for r in rows: grouped[str(items[r['id']][key])].append(r)
            groups[key] = {k:summary(v) for k,v in grouped.items()}
        group_means = [v['mean_pct'] for v in groups['physics_group'].values() if v['mean_pct'] is not None]
        methods[mode] = {**summary(rows), 'physics_group_mean_pct':float(np.mean(group_means)), 'groups':groups}
        records.extend([{**r,'mode':mode} for r in rows])
    changed = sum(old[i]['predictions']['centred'].get('frames') != new[i]['predictions']['centred'].get('frames') for i in old)
    out = {'scope':'full inspected development benchmark, not held out; no item exclusions by error',
           'fps':a.fps, 'protocol':'identical input observations and velocity constraints; contact logic is the sole factor, actual fitted frame counts may differ',
           'changed_frame_counts':changed, 'methods':methods, 'records':records}
    Path(f'docs/phyediting/contact_comparison{a.fps}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:{x:v for x,v in m.items() if x!='groups'} for k,m in methods.items()},indent=2))

if __name__ == '__main__':
    main()
