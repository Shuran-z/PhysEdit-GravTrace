"""Diagnostic shared-gravity fit across disjoint flights; selected tails plus hash controls."""
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scripts.phyediting_gen import _sample,_read_tracks
from scripts.phyediting_joint_trial import joint_fit
from scripts.phyediting_joint_coverage import observed_frames


def evaluate(job):
    meta,samples=job
    return {**meta,'joint':joint_fit(samples)}


def main():
    read=lambda p:list(map(json.loads,open(p)))
    audit=json.loads(Path('docs/phyediting/cross_window_coverage15.json').read_text())['records']
    eligible=[r for r in audit if r['selected'] and r['primary_cause']=='at_least_four']
    tails=[r for r in eligible if r['primary_tail_over10']]
    controls=sorted([r for r in eligible if not r['primary_tail_over10']],key=lambda r:hashlib.sha256(('cross-window-v1:'+r['id']).encode()).hexdigest())[:8]
    selected=tails+controls
    items={r['id']:r for r in read('runs/benchmark_gravity_v1/items.jsonl')}
    oracle={r['id']:r for r in read('runs/all_oracle.jsonl')}
    tracks=_read_tracks('runs/tracks_all.jsonl');primary_tracks=_read_tracks('runs/genfloor_tracks_15.jsonl')
    rows={r['item_id']:r for r in read('runs/genfloor_rows_15.jsonl')}
    baseline={r['id']:r['predictions']['centred'] for r in read('runs/contact_off15_v1.jsonl')}
    jobs=[]
    for record in selected:
        it=items[record['id']];aux=oracle[record['selected']];w=aux['window']
        tr=tracks[it['trajectory']+'__'+it['camera_id']]['tracks'][it['object_name']]
        pairs=observed_frames(tr,w['start_frame'],w['start_frame']+w['max_frames']-1)
        sample={k:aux[k] for k in ['id','scenario','fps','image_size','object']}
        motion=dict(aux['motion']);motion.pop('v0',None);motion.pop('directions',None)
        motion.update(speed=[0.,8.],angle_deg=[-89.,89.],t0=[(pairs[0][0]-w['start_frame'])/aux['fps']]*2)
        sample.update(motion=motion,camera=it['camera'],offset_mode='centred',
            boxes={'frames':[f for f,b in pairs],'xyxy':[b for f,b in pairs],'times':[(f-pairs[0][0])/aux['fps'] for f,b in pairs]},
            window={'start_frame':0,'max_frames':len(pairs),'contact_check':False,'edges_visible':{str(f):w.get('edges_visible',{}).get(str(f),[True]*4) for f,b in pairs}})
        row=rows[it['id']];primary=_sample(row,it,primary_tracks[row['sample_id']],'agnostic',{})
        primary['offset_mode']='centred';primary['window']['contact_check']=False
        jobs.append(({'id':it['id'],'cohort':'selected_tail' if record['primary_tail_over10'] else 'hash_control'},[primary,sample]))
    records=[]
    with ProcessPoolExecutor(max_workers=6) as pool,open('runs/cross_window_pilot15.jsonl','w') as f:
        for r in pool.map(evaluate,jobs):
            r['target']=items[r['id']]['gravity'];r['baseline']=baseline[r['id']]
            records.append(r);f.write(json.dumps(r)+'\n');f.flush()
    summary={}
    for cohort in ['all','selected_tail','hash_control']:
        rs=[r for r in records if cohort=='all' or r['cohort']==cohort];summary[cohort]={}
        for method in ['baseline','joint']:
            es=[abs(r[method]['gravity']/r['target']-1)*100 for r in rs if r[method]['status']=='ok']
            summary[cohort][method]={'attempted':len(rs),'ok':len(es),'failed_or_bound':len(rs)-len(es),'mean_pct':float(np.mean(es)) if es else None,'max_pct':max(es) if es else None}
    result={'scope':'selected 4 eligible high-error development tails plus 8 fixed-hash controls; not overall or unseen performance',
            'caveat':'extra disjoint-window declaration and real observations; independent velocity per window, fixed independent-fit direction hypotheses; not information matched to prior single-window baselines',
            'summary':summary,'records':records}
    Path('docs/phyediting/cross_window_pilot15.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
