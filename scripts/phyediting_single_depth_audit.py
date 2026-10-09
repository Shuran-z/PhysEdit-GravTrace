"""Audit cached single-image inference inputs without filtering by target error."""
import json
from pathlib import Path
from collections import Counter

MODELS=['depthpro','dav2_metric','zoedepth','moge2','unidepth_v2']


def input_matches(sample,window,cached_K):
    width,height=sample['image_size'];cam=sample['camera']
    K=[[cam['fx']*width,0,cam.get('cx',.5)*width],[0,abs(cam['fy'])*height,cam.get('cy',.5)*height],[0,0,1]]
    pairs=dict(zip(window['frames'],window['boxes']))
    return K==cached_K and all(pairs.get(f)==b for f,b in zip(sample['boxes']['frames'],sample['boxes']['xyxy']))


def main():
    read=lambda p:list(map(json.loads,open(p)))
    samples=read('runs/unknown_common.jsonl')
    old={w['id']:(w,j) for j in read('runs/final_depth_jobs.jsonl') for w in j['windows']}
    items={r['id']:r for r in read('runs/benchmark_gravity_v1/items.jsonl')}
    depths={m:{r['id']:r for r in read(f'runs/final_depth_{m}.jsonl')} for m in MODELS}
    rows=[]
    for s in samples:
        prior=old.get(s['id'])
        matched=bool(prior and input_matches(s,prior[0],prior[1]['K']))
        video=bool(prior and prior[1]['video'].endswith('/'+items[s['id']]['video']))
        record={'id':s['id'],'input_matches':matched,'video_matches':video,'models':{}}
        for model in MODELS:
            r=depths[model].get(s['id'])
            valid=bool(r and r['model']==model and len(r['frames'])==len(set(r['frames'])) and len(r['depth'])==len(r['frames']) and set(s['boxes']['frames']).issubset(r['frames']))
            record['models'][model]='matched' if matched and video and valid else 'requires_review'
        rows.append(record)
    result={'scope':'single-image cache input audit; frames are inferred independently, extra cached frames do not inform retained-frame predictions; no quality/error filtering',
            'eligible':len(samples),'input_matches':sum(r['input_matches'] for r in rows),'video_matches':sum(r['video_matches'] for r in rows),
            'models':{m:dict(Counter(r['models'][m] for r in rows)) for m in MODELS},'records':rows}
    Path('docs/phyediting/single_depth_input_audit.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print({k:v for k,v in result.items() if k!='records'})

if __name__=='__main__':main()
