"""Audit disjoint free-flight observations of the same object; no fitting/error selection."""
import json
from pathlib import Path
from collections import defaultdict,Counter
from scripts.phyediting_joint_coverage import observed_frames


def main():
    read=lambda p:list(map(json.loads,open(p)))
    items=read('runs/benchmark_gravity_v1/items.jsonl')
    tracks={r['id']:r for r in read('runs/tracks_all.jsonl')}
    windows=defaultdict(list)
    for r in read('runs/all_oracle.jsonl'):
        if r['scenario']=='projectile':
            name=r['id'].rsplit('__',2)[-2]
            windows[(r['gt_id'],r['meta']['camera'],name)].append(r)
    sampling={r['id']:r for r in json.loads(Path('docs/phyediting/sampling_audit15.json').read_text())['records']}
    baseline={r['id']:r['predictions']['centred'] for r in read('runs/contact_off15_v1.jsonl')}
    rows=[];jobs=[]
    for it in items:
        start=it['window']['start_frame'];end=start+it['window']['max_frames']-1
        video=it['trajectory']+'__'+it['camera_id']
        tr=tracks.get(video,{}).get('tracks',{}).get(it['object_name'])
        candidates=[]
        if tr:
            for aux in windows[(it['trajectory'],it['camera_id'],it['object_name'])]:
                w=aux['window'];lo=w['start_frame'];hi=lo+w['max_frames']-1
                if not(hi<start or lo>end):continue
                pairs=observed_frames(tr,lo,hi)
                seen=w.get('edges_visible',{})
                if sum(sum(seen.get(str(f),[True]*4))>=2 and min(b[2]-b[0],b[3]-b[1])>=6 for f,b in pairs)<4:continue
                candidates.append((len(pairs),aux['id'],pairs,aux))
        candidates.sort(key=lambda r:(-r[0],r[1]))
        pred=baseline[it['id']]
        # Labels only added after candidates are fixed; never used for ranking.
        tail=pred.get('status')=='ok' and abs(pred['gravity']/it['gravity']-1)>.1
        rows.append({'id':it['id'],**{k:it[k] for k in ['event','gravity','camera_id','physics_group']},
                     'primary_cause':sampling[it['id']]['cause'],'primary_tail_over10':tail,
                     'disjoint_candidates':len(candidates),'selected':candidates[0][1] if candidates else None})
        if candidates:
            _,_,pairs,aux=candidates[0]
            jobs.append({'item_id':it['id'],'auxiliary_id':aux['id'],'frames':[f for f,b in pairs]})
    def stats(rs):return {'attempted':len(rs),'with_disjoint':sum(bool(r['selected']) for r in rs)}
    groups={}
    for key in ['event','gravity','camera_id','physics_group','primary_cause','primary_tail_over10']:
        bins=defaultdict(list)
        for r in rows:bins[str(r[key])].append(r)
        groups[key]={k:stats(v) for k,v in bins.items()}
    result={'scope':'15fps existing SAM2 tracks, same-object disjoint simulator-declared free flights, fixed longest-frame-count then ID selection',
            'caveat':'coverage only; extra window pose/state/visibility and observations would need matched baselines. Gravity/error labels are only attached after selection. No claim of improved inversion.',
            **stats(rows),'groups':groups,'records':rows}
    Path('docs/phyediting/cross_window_coverage15.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    Path('runs/cross_window_auxiliary15.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in jobs))
    print(stats(rows));print(groups['primary_cause']);print(groups['primary_tail_over10'])

if __name__=='__main__':main()
