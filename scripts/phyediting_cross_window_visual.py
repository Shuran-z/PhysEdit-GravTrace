"""Render existing selected auxiliary diagnostics; visual QA, no changed predictions."""
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.phyediting_cross_window_trial import build_jobs
from scripts.phyediting_cross_window_diagnosis import project_sample


def main():
    jobs,items,_=build_jobs();windows={r['id']:r for r in map(json.loads,open('runs/all_oracle.jsonl'))}
    labels={r['id']:r for r in json.loads(Path('docs/phyediting/cross_window_diagnosis15.json').read_text())['records']}
    rows=[];manifest=[]
    for meta,samples in jobs:
        if labels[meta['id']]['independent_observed_errors_pct'][1]<=60:continue
        it=items[meta['id']];s=samples[1];start=windows[s['id']]['window']['start_frame']
        oracle,audit=project_sample(s,it['trajectory'],it['object_name'],start)
        frames=audit['reference_frames'];indices=sorted(set([0,len(frames)//2,len(frames)-1]));panels=[]
        cap=cv2.VideoCapture(str(Path('data/phyediting')/it['video']))
        for k in indices:
            f=frames[k];cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read()
            if not ok:raise RuntimeError('unreadable video frame')
            b=s['boxes']['xyxy'][k];g=oracle['boxes']['xyxy'][k]
            for box,color in [(b,(0,0,255)),(g,(0,255,0))]:
                x0,y0,x1,y1=np.rint(box).astype(int);cv2.rectangle(img,(x0,y0),(x1,y1),color,2)
            img=cv2.resize(img,(640,360));cv2.putText(img,f'{it["event"]} frame {f}: red SAM2 / green projected',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            panels.append(img)
        cap.release();rows.append(np.hstack(panels));manifest.append({'id':meta['id'],'auxiliary':s['id'],'frames':[frames[k] for k in indices]})
    cv2.imwrite('runs/cross_window_visual15.jpg',np.vstack(rows))
    Path('docs/phyediting/cross_window_visual15.json').write_text(json.dumps({'scope':'three existing error-selected auxiliary cases, diagnostic only','records':manifest},indent=2)+'\n')

if __name__=='__main__':main()
