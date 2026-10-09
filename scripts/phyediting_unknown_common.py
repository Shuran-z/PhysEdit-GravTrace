"""Prepare the existing common frames with velocity truth removed; no new exclusions."""
import json
from pathlib import Path


def remove_velocity(s):
    motion=dict(s['motion']);motion.pop('v0',None);motion.pop('directions',None)
    motion.update(speed=[0.,8.],angle_deg=[-89.,89.])
    return {**s,'motion':motion,'offset_mode':'centred',
            'window':{**s['window'],'contact_check':False,'depth_require_all_frames':True,'depth_anchor_extrapolate':True}}


def main():
    with open('runs/unknown_common.jsonl','w') as out:
        for s in map(json.loads,open('runs/common_sam2.jsonl')):
            out.write(json.dumps(remove_velocity(s))+'\n')
    Path('runs/unknown_common.jsonl.coverage.json').write_text(Path('runs/common_sam2.jsonl.coverage.json').read_text())

if __name__=='__main__':main()
