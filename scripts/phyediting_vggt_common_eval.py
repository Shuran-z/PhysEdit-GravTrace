"""Validate exact common-frame VGGT records and refit without reference velocity."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scripts.phyediting_bounded_depth import lift_job, evaluate, summarize


def merge_records(samples, reused, rerun):
    records=reused+rerun
    by_id={r['id']:r for r in records}
    expected={s['id'] for s in samples}
    if len(by_id)!=len(records) or set(by_id)!=expected:
        raise ValueError('duplicate, missing or unexpected depth IDs')
    for s in samples:
        r=by_id[s['id']]
        if r['model']!='vggt' or r['frames']!=s['boxes']['frames'] or len(r['depth'])!=len(r['frames']):
            raise ValueError('depth model, ordered frames or length mismatch')
    return by_id


def main():
    read=lambda p:list(map(json.loads,open(p)))
    samples=read('runs/unknown_common.jsonl')
    depths=merge_records(samples,read('runs/vggt_common_reused.jsonl'),read('runs/vggt_common_rerun.jsonl'))
    Path('runs/vggt_common_merged.jsonl').write_text(''.join(json.dumps(depths[s['id']])+'\n' for s in samples))
    jobs=[]
    for s in samples:
        data,error=lift_job(s,depths[s['id']])
        jobs.append(({'id':s['id'],'model':'vggt'},error if error else data))
    records=[]
    with ProcessPoolExecutor(max_workers=6) as pool,open('runs/vggt_common_bounded.jsonl','w') as out:
        for r in pool.map(evaluate,jobs):
            records.append(r);out.write(json.dumps(r)+'\n');out.flush()
    items={r['id']:r for r in read('runs/benchmark_gravity_v1/items.jsonl')}
    result={'scope':'VGGT common neural-input and fitting frames, development set, unknown initial velocity',
            'input_audit':{'eligible':len(samples),'reused':534,'rerun':502},
            'caveat':'Depth lifting uses metre residuals and centre ballistic motion; geometry/spin/drag representation differs. No independent test or global optimality guarantee.',
            'method':summarize(records,items)['vggt']}
    Path('docs/phyediting/vggt_common_result.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print({k:v for k,v in result['method'].items() if k!='groups'})

if __name__=='__main__':main()
