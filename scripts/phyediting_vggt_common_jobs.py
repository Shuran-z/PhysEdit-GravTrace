"""Audit exact VGGT inference inputs and queue only changed common-frame windows."""
import argparse
import json
from pathlib import Path


def same_input(new, old, new_K, old_K):
    return (new['frames']==old['frames'] and new['boxes']==old['boxes'] and new_K==old_K)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-root',required=True);a=p.parse_args()
    old={w['id']:(w,j) for j in map(json.loads,open('runs/final_depth_jobs.jsonl')) for w in j['windows']}
    depths={r['id']:r for r in map(json.loads,open('runs/final_depth_vggt.jsonl'))}
    samples=list(map(json.loads,open('runs/unknown_common.jsonl')))
    items={r['id']:r for r in map(json.loads,open('runs/benchmark_gravity_v1/items.jsonl'))}
    jobs=[];reused=[];audit=[]
    for s in samples:
        width,height=s['image_size'];cam=s['camera']
        K=[[cam['fx']*width,0,cam.get('cx',.5)*width],[0,abs(cam['fy'])*height,cam.get('cy',.5)*height],[0,0,1]]
        w={'id':s['id'],'frames':s['boxes']['frames'],'boxes':s['boxes']['xyxy']}
        previous=old.get(s['id']);rec=depths.get(s['id'])
        reuse=bool(previous and rec and same_input(w,previous[0],K,previous[1]['K']) and rec['frames']==w['frames'])
        if reuse:reused.append(rec)
        else:jobs.append({'video':a.video_root.rstrip('/')+'/'+items[s['id']]['video'],'K':K,'windows':[w]})
        audit.append({'id':s['id'],'reuse':reuse,'frames':len(w['frames'])})
    Path('runs/vggt_common_jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))
    Path('runs/vggt_common_reused.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reused))
    out={'scope':'VGGT exact input audit: ordered frames, sampling boxes, intrinsics; reuse does not filter by depth validity or gravity error',
         'eligible':len(samples),'reuse':len(reused),'rerun':len(jobs),'records':audit}
    Path('docs/phyediting/vggt_input_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print({k:v for k,v in out.items() if k!='records'})

if __name__=='__main__':main()
