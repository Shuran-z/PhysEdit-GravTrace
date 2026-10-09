"""Development image-only motion-box refinement at unchanged observation timestamps."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phyediting_gen import _read_tracks, _sample
from gravtrace.fit import safe_fit


def refine_box(image, background, box, threshold=12., margin=8):
    height,width=image.shape[:2]
    x0,y0,x1,y1=np.asarray(box,float)
    lo=np.maximum(np.floor([x0-margin,y0-margin]).astype(int),0)
    hi=np.minimum(np.ceil([x1+margin,y1+margin]).astype(int),[width,height])
    if (hi<=lo).any():return list(box),'empty_roi'
    crop=image[lo[1]:hi[1],lo[0]:hi[0]].astype(float)
    bg=background[lo[1]:hi[1],lo[0]:hi[0]].astype(float)
    difference=np.max(np.abs(crop-bg),axis=2)
    mask=(difference>threshold).astype(np.uint8)
    n,labels,stats,centres=cv2.connectedComponentsWithStats(mask,8)
    candidates=[]
    for k in range(1,n):
        x,y,w,h,area=stats[k]
        if ((x==0 and lo[0]>0) or (y==0 and lo[1]>0) or
            (x+w==mask.shape[1] and hi[0]<width) or (y+h==mask.shape[0] and hi[1]<height)):
            continue  # an ROI-cut component cannot supply an object boundary
        candidate=np.array([x+lo[0],y+lo[1],x+w+lo[0],y+h+lo[1]],float)
        inter=np.maximum(np.minimum(candidate[2:],[x1,y1])-np.maximum(candidate[:2],[x0,y0]),0).prod()
        if area<36 or inter<.3*max((x1-x0)*(y1-y0),1):continue
        if np.max(np.abs(candidate-np.array(box)))>margin:continue
        if min(w,h)<6:continue
        candidates.append((inter,candidate.tolist()))
    if not candidates:return list(box),'no_component'
    return max(candidates,key=lambda v:v[0])[1],'refined'


def evaluate(job):
    meta,sample,path=job
    sample=json.loads(json.dumps(sample))
    t=np.asarray(sample['boxes']['times']);start=meta['start_frame']
    frames=np.rint(start+t*30).astype(int)
    cap=cv2.VideoCapture(str(path));images=[]
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,image=cap.read()
        if not ok:
            cap.release();return {**meta,'prediction':{'status':'video_read_error'},'refined_frames':0}
        images.append(image)
    cap.release()
    # Background uses only the same observed frames, not simulator pixels or extra times.
    background=np.median(np.stack(images),axis=0)
    reasons={};boxes=[]
    for image,box in zip(images,sample['boxes']['xyxy']):
        refined,reason=refine_box(image,background,box)
        boxes.append(refined);reasons[reason]=reasons.get(reason,0)+1
    sample['boxes']['xyxy']=boxes
    pred=safe_fit(sample)
    return {**meta,'prediction':pred,'frame_reasons':reasons,'refined_frames':reasons.get('refined',0),
            'frames':frames.tolist(),'refined_boxes':boxes}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    items={r['id']:r for r in map(json.loads,open('runs/benchmark_gravity_v1/items.jsonl'))}
    rows={r['item_id']:r for r in map(json.loads,open('runs/genfloor_rows_15.jsonl'))}
    tracks=_read_tracks('runs/genfloor_tracks_15.jsonl')
    baseline={r['id']:r['predictions']['centred'] for r in map(json.loads,open('runs/contact_off15_v1.jsonl'))}
    selected=json.load(open('docs/phyediting/gravity_profile15.json'))['records']
    jobs=[]
    for r in selected:
        i=r['id'];it,row=items[i],rows[i]
        sample=_sample(row,it,tracks.get(row['sample_id'],{}),'agnostic',{})
        sample['offset_mode']='centred';sample['window']['contact_check']=False
        jobs.append(({'id':i,'cohort':r['cohort'],'start_frame':it['window']['start_frame']},sample,Path('data/phyediting')/it['video']))
    records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(evaluate,jobs):
            r['gravity_target']=items[r['id']]['gravity'];r['baseline']=baseline[r['id']];records.append(r)
    summary={}
    for cohort in ['all','error_selected_tail','hash_control']:
        for mode in ['baseline','prediction']:
            rs=[r for r in records if cohort=='all' or r['cohort']==cohort]
            errors=[abs(r[mode]['gravity']/r['gravity_target']-1)*100 for r in rs if r[mode]['status']=='ok']
            summary[cohort+'_'+mode]={'attempted':len(rs),'ok':len(errors),'failed_or_bound':len(rs)-len(errors),
                                    'mean_pct':float(np.mean(errors)) if errors else None,'max_pct':max(errors) if errors else None}
    out={'scope':'19 development diagnostic cases selected for prior gravity profile; not overall performance or held out',
         'protocol':'same actual timestamps and unknown velocity; image median background and connected motion component inside fixed SAM2 ROI; threshold12, margin8, fallback to original box on image-only gate failure; no oracle box input',
         'summary':summary,'records':records}
    Path('docs/phyediting/motion_refine15.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
