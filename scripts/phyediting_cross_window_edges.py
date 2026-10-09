"""Compare video-observable smoothness with diagnostic oracle edge bias; no prediction changes."""
import json
from pathlib import Path
import numpy as np
from scripts.phyediting_cross_window_trial import build_jobs
from scripts.phyediting_cross_window_diagnosis import project_sample
from scripts.phyediting_edge_diagnosis import edge_stats


def observable_features(sample):
    b=np.asarray(sample['boxes']['xyxy'],float);t=np.asarray(sample['boxes']['times'],float)
    u=(t-t.mean())/max(np.ptp(t),1e-9);A=np.column_stack([np.ones(len(t)),u,u*u])
    residual=b-A@np.linalg.lstsq(A,b,rcond=None)[0]
    width,height=sample['image_size'];limits=np.array([width,height,width,height])
    return {'frames':len(t),'duration_s':float(np.ptp(t)),
            'quadratic_residual_rms_px':np.sqrt(np.mean(residual**2,axis=0)).tolist(),
            'border_frames':int(np.any((b<=1)|(b>=limits-1),axis=1).sum()),
            'tiny_frames':int(((b[:,2:]-b[:,:2]).min(axis=1)<6).sum())}


def main():
    jobs,items,_=build_jobs();oracle={r['id']:r for r in map(json.loads,open('runs/all_oracle.jsonl'))}
    diagnoses={r['id']:r for r in json.loads(Path('docs/phyediting/cross_window_diagnosis15.json').read_text())['records']}
    records=[]
    for meta,samples in jobs:
        it=items[meta['id']]
        for k,s in enumerate(samples):
            features=observable_features(s)
            start=it['window']['start_frame'] if k==0 else oracle[s['id']]['window']['start_frame']
            projected,audit=project_sample(s,it['trajectory'],it['object_name'],start)
            truth=np.asarray(projected['boxes']['xyxy']);observed=np.asarray(s['boxes']['xyxy']);t=np.asarray(s['boxes']['times'])
            seen=s['window'].get('edges_visible',{});visibility=np.array([seen.get(str(f),[True]*4) for f in s['boxes']['frames']],bool)
            width,height=s['image_size'];limits=np.array([width,height,width,height])
            usable=visibility&(truth>1)&(truth<limits-1)&(observed>1)&(observed<limits-1)
            edges={name:edge_stats(t[usable[:,j]],(observed-truth)[usable[:,j],j]) for j,name in enumerate(['left','top','right','bottom'])}
            records.append({**meta,'role':'primary' if k==0 else 'auxiliary','window_id':s['id'],
                            'observable':features,'declared_hidden_edge_count':int((~visibility).sum()),
                            'oracle_audit':audit,'diagnostic_edges':edges,
                            'independent_error_pct':diagnoses[meta['id']]['independent_observed_errors_pct'][k]})
    result={'scope':'same 12 selected development pairs; observable features computed without truth; oracle edge residuals and error labels diagnostic only',
            'caveat':'quadratic image-box smoothness is not a calibrated segmentation-quality measure and can reflect spin or perspective. No thresholds, frame rejection or prediction changes.',
            'records':records}
    Path('docs/phyediting/cross_window_edges15.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in sorted([r for r in records if r['role']=='auxiliary'],key=lambda r:r['independent_error_pct'] or -1,reverse=True):
        print(r['window_id'],r['independent_error_pct'],max(r['observable']['quadratic_residual_rms_px']),r['oracle_audit']['tiny_or_offscreen_frames'],r['declared_hidden_edge_count'],max(e.get('curvature_excursion_px',0) for e in r['diagnostic_edges'].values()))

if __name__=='__main__':main()
