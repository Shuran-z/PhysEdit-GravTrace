"""Audit added disjoint windows under the existing fully-visible common-frame rule."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scripts.phyediting_cross_window_trial import build_jobs


def main():
    jobs,items,_=build_jobs();records=[]
    for meta,samples in jobs:
        s=samples[1];b=np.asarray(s['boxes']['xyxy']);w,h=s['image_size'];limits=np.array([w,h,w,h])
        seen=s['window'].get('edges_visible',{});v=np.array([seen.get(str(f),[True]*4) for f in s['boxes']['frames']],bool)
        mask=np.all(v&(b>1)&(b<limits-1),axis=1)
        records.append({**meta,'added_window':s['id'],'frames':len(mask),'common_frames':int(mask.sum()),'eligible':bool(mask.sum()>=4),
                        'retained_frame_ids':[f for f,k in zip(s['boxes']['frames'],mask) if k]})
    out={'scope':'existing selected 12-case pilot, auxiliary admission audit only; main windows and all items retained, no fitting or target-based selection',
         'rule':'all four declared edges visible and observed box >1px from every border, at least four actual frames; uses the existing common-frame protocol',
         'attempted':len(records),'with_eligible_auxiliary':sum(r['eligible'] for r in records),'records':records}
    Path('docs/phyediting/auxiliary_common_audit15.json').write_text(json.dumps(out,indent=2)+'\n')
    print([(r['cohort'],r['common_frames'],r['eligible']) for r in records])

if __name__=='__main__':main()
