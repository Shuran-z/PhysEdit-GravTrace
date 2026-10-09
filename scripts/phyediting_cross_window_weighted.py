"""Fixed residual-RMS window scaling control; same observations, no truth-derived weights."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scripts.phyediting_cross_window_trial import build_jobs
from scripts.phyediting_joint_trial import joint_fit


def evaluate(job):
    meta,samples=job
    return {**meta,'weighted':joint_fit(samples,noise_weighted=True)}


def main():
    jobs,items,baseline=build_jobs()
    old={r['id']:r for r in json.loads(Path('docs/phyediting/cross_window_pilot15.json').read_text())['records']}
    with ProcessPoolExecutor(max_workers=6) as pool:records=list(pool.map(evaluate,jobs))
    for r in records:r.update(target=items[r['id']]['gravity'],baseline=baseline[r['id']],unweighted=old[r['id']]['joint'])
    summary={}
    for cohort in ['all','selected_tail','hash_control']:
        rs=[r for r in records if cohort=='all' or r['cohort']==cohort];summary[cohort]={}
        for method in ['baseline','unweighted','weighted']:
            es=[abs(r[method]['gravity']/r['target']-1)*100 for r in rs if r[method]['status']=='ok']
            summary[cohort][method]={'attempted':len(rs),'ok':len(es),'failed_or_bound':len(rs)-len(es),'mean_pct':float(np.mean(es)) if es else None,'max_pct':max(es) if es else None}
    result={'scope':'same selected development 12-case pilot; no independent validation',
            'rule':'window residuals divided by max(1px, RMS of all residuals at independent unknown-velocity fit); all observations retained; scales fixed before joint optimization',
            'caveat':'residual scales are not calibrated uncertainties; bias may be absorbed by gravity/velocity. Extra window declarations remain unmatched to prior single-window baselines. Default unchanged.',
            'summary':summary,'records':records}
    Path('docs/phyediting/cross_window_weighted15.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
