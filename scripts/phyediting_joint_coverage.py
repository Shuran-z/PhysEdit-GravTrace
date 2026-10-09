"""Audit concurrent auxiliary flight tracks without gravity/error-based selection."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def observed_frames(track, start, end, fps=15):
    step=30//fps
    return [(f,b) for f,b in zip(track['frames'],track['xyxy'])
            if start <= f <= end and f>=23 and (f-23)%step==0]


def main():
    items=list(map(json.loads,open('runs/benchmark_gravity_v1/items.jsonl')))
    tracks={r['id']:r for r in map(json.loads,open('runs/tracks_all.jsonl'))}
    oracle=defaultdict(list)
    for r in map(json.loads,open('runs/all_oracle.jsonl')):
        if r['scenario']=='projectile':oracle[(r['gt_id'],r['meta']['camera'])].append(r)
    rows=[];jobs=[]
    for it in items:
        key=(it['trajectory'],it['camera_id']);video='__'.join(key)
        record=tracks.get(video,{});available=record.get('tracks',{})
        w=it['window'];start=w['start_frame'];end=start+w['max_frames']-1
        candidates=[]
        for aux in oracle[key]:
            name=aux['id'].rsplit('__',2)[-2]
            if name==it['object_name'] or name not in available:continue
            aw=aux['window'];lo=max(start,aw['start_frame']);hi=min(end,aw['start_frame']+aw['max_frames']-1)
            pairs=observed_frames(available[name],lo,hi)
            if len(pairs)<4:continue
            seen=aw.get('edges_visible',{})
            useful=sum(sum(seen.get(str(f),[True]*4))>=2 and min(b[2]-b[0],b[3]-b[1])>=6 for f,b in pairs)
            if useful<4:continue
            candidates.append((len(pairs),aux['id'],name,pairs,aux))
        candidates.sort(key=lambda r:(-r[0],r[1]))
        rows.append({'id':it['id'], **{k:it[k] for k in ['event','camera_id','gravity','physics_group']},
                     'concurrent_auxiliary_windows':len(candidates),
                     'auxiliary_id':candidates[0][1] if candidates else None})
        if candidates:
            _,_,name,pairs,aux=candidates[0]
            # Original declared start state remains; only sampled timestamps change.
            sample={k:aux[k] for k in ['id','scenario','fps','image_size','object','motion']}
            sample['motion']=dict(sample['motion']);sample['motion'].pop('v0',None)
            sample['motion'].update(speed=[0.,8.],angle_deg=[-89.,89.]);sample['motion'].pop('directions',None)
            sample['motion']['t0']=[(pairs[0][0]-aux['window']['start_frame'])/30]*2
            sample['camera']=it['camera'];sample['offset_mode']='centred'
            sample['boxes']={'frames':[f for f,b in pairs],'xyxy':[b for f,b in pairs], 'times':[(f-pairs[0][0])/30 for f,b in pairs]}
            sample['window']={'start_frame':0,'max_frames':len(pairs),'contact_check':False,
                              'edges_visible':{str(f):aux['window']['edges_visible'].get(str(f),[True]*4) for f,b in pairs}}
            jobs.append({'item_id':it['id'],'auxiliary':sample})
    groups={}
    for key in ['event','camera_id','gravity','physics_group']:
        totals=Counter(str(r[key]) for r in rows);usable=Counter(str(r[key]) for r in rows if r['concurrent_auxiliary_windows'])
        groups[key]={k:{'attempted':n,'with_auxiliary':usable[k]} for k,n in totals.items()}
    out={'scope':'audit on inspected benchmark, 15fps original SAM2 tracks; no error or gravity quality gate',
         'caveat':'auxiliary flight windows, poses and visibility remain simulator-declared; extra object observations must also be provided to baselines',
         'attempted':len(rows),'with_auxiliary':len(jobs),'groups':groups,'records':rows}
    Path('docs/phyediting/joint_coverage15.json').write_text(json.dumps(out,indent=2)+'\n')
    Path('runs/joint_auxiliary15.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))
    print(json.dumps({k:v for k,v in out.items() if k not in ('groups','records')},indent=2))

if __name__=='__main__':main()
