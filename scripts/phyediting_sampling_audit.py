"""Count real 15fps observations and cadence capacity, without fitting or dropping items."""
import json
import math
from collections import Counter,defaultdict
from pathlib import Path


def main():
    read=lambda p:list(map(json.loads,open(p)))
    items={r['id']:r for r in read('runs/benchmark_gravity_v1/items.jsonl')}
    tracks={r['id']:r for r in read('runs/genfloor_tracks_15.jsonl')}
    records=[]
    for row in read('runs/genfloor_rows_15.jsonl'):
        it=items[row['item_id']];fps=row['output_fps'];ref=it['fps'];cond=row['condition_frame_index']/ref
        start=it['window']['start_frame']/ref;end=start+(it['window']['max_frames']-1)/ref
        first=max(0,math.ceil((start-cond)*fps-1e-6));last=math.floor((end-cond)*fps+1e-6)
        capacity=max(0,last-first+1)
        tr=tracks.get(row['sample_id'],{}).get('tracks',{}).get(it['object_name'],{})
        available=sum(first<=f<=last for f in tr.get('frames',[]))
        cause='cadence_below_four' if capacity<4 else ('missing_track_frames' if available<4 else 'at_least_four')
        records.append({'id':it['id'],'capacity':capacity,'available':available,'cause':cause})
    def stats(rows):return {'attempted':len(rows),'causes':dict(Counter(r['cause'] for r in rows))}
    groups={}
    for key in ['event','gravity','camera_id','physics_group']:
        bins=defaultdict(list)
        for r in records:bins[str(items[r['id']][key])].append(r)
        groups[key]={k:stats(v) for k,v in bins.items()}
    result={'scope':'15fps declared windows; cadence capacity versus recorded real track frames; no interpolation, no parameter fitting or error filtering',**stats(records),'groups':groups,'records':records}
    Path('docs/phyediting/sampling_audit15.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(stats(records))

if __name__=='__main__':main()
