"""Run the bounded unknown-velocity 2D control on existing identical fit frames."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scripts.baselines import camera_frame,projection_jacobian
from scripts.phyediting_bounded_baseline import bounded_motion


def add_groups(out,records):
    items={r['id']:r for r in map(json.loads,open('runs/benchmark_gravity_v1/items.jsonl'))}
    predictions={r['id']:r for r in records}
    groups={}
    for key in ['event','gravity','camera_id','physics_group']:
        bins=defaultdict(list)
        for i,it in items.items():bins[str(it[key])].append(predictions.get(i,{'status':'no_common_frames'}))
        groups[key]={}
        for k,rs in bins.items():
            es=[abs(r['gravity']/r['gravity_target']-1)*100 for r in rs if r['status']=='ok']
            groups[key][k]={'attempted':len(rs),'ok':len(es),'failed':len(rs)-len(es),
                            'mean_pct':float(np.mean(es)) if es else None,'max_pct':max(es) if es else None}
    out['groups']=groups


def main():
    records=[]
    for sample in map(json.loads,open('runs/unknown_common.jsonl')):
        boxes=np.asarray(sample['boxes']['xyxy']);frames=np.asarray(sample['boxes']['frames'])
        t=(frames-sample['window']['start_frame'])/sample['fps']
        R,T,K=camera_frame(sample);position=R@np.array(sample['object']['position'])+T
        J=projection_jacobian(position,K)@R
        pred=bounded_motion(t,.5*(boxes[:,:2]+boxes[:,2:]),J,sample['motion']['gravity_dir'])
        records.append({'id':sample['id'],**pred,'gravity_target':sample['truth']['gravity']})
    Path('runs/bounded_2d.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    errors=[abs(r['gravity']/r['gravity_target']-1)*100 for r in records if r['status']=='ok']
    out={'scope':'bounded 2D common-frame development control; not held out',
         'bounds':'speed0-8m/s, elevation+-89deg, gravity0.1-80; direction free, no velocity truth',
         'caveat':'linear projection and parabolic centre model; geometry/spin/drag representations differ from ours; SLSQP finite starts do not prove global optimality',
         'input':1038,'common_eligible':len(records),'ok':len(errors),'failed_or_missing':1038-len(errors),
         'mean_pct':float(np.mean(errors)) if errors else None,'max_pct':max(errors) if errors else None}
    add_groups(out,records)
    Path('docs/phyediting/bounded_2d.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
