"""Development-only edge-error diagnosis; oracle geometry is never fed to the fitter."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phyediting_gen import _read_tracks, _sample
from gravtrace.camera import Camera, quat_to_matrix


def edge_stats(t, error):
    t, error = np.asarray(t, float), np.asarray(error, float)
    if len(t) < 4 or np.ptp(t) == 0:
        return {"frames": len(t)}
    u = (t - t.mean()) / np.ptp(t)
    linear = np.column_stack([np.ones(len(u)), u])
    curve = u ** 2 - linear @ np.linalg.lstsq(linear, u ** 2, rcond=None)[0]
    detrended = error - linear @ np.linalg.lstsq(linear, error, rcond=None)[0]
    coefficient = float(curve @ detrended / (curve @ curve))
    correlation = (float(np.corrcoef(detrended[:-1], detrended[1:])[0, 1])
                   if np.std(detrended[:-1]) > 1e-9 and np.std(detrended[1:]) > 1e-9 else None)
    return {"frames": len(t), "offset_px": float(np.median(error)),
            "range_after_offset_px": float(np.ptp(error)),
            "linear_detrended_rms_px": float(np.sqrt(np.mean(detrended ** 2))),
            "curvature_coefficient_px": coefficient,
            "curvature_excursion_px": float(abs(coefficient) * np.ptp(curve)),
            "lag1_after_linear_detrend": correlation}


def main():
    items = {r['id']: r for r in map(json.loads, open('runs/benchmark_gravity_v1/items.jsonl'))}
    selected = json.load(open('docs/phyediting/tail_diagnosis.json'))['records']
    result = []
    for fps in (30, 15):
        rows = {r['item_id']: r for r in map(json.loads, open(f'runs/genfloor_rows_{fps}.jsonl'))}
        tracks = _read_tracks(f'runs/genfloor_tracks_{fps}.jsonl')
        for rec in [r for r in selected if r['fps'] == fps]:
            it, row = items[rec['id']], rows[rec['id']]
            sample = _sample(row, it, tracks.get(row['sample_id'], {}), 'agnostic', {})
            n = min(len(sample['boxes']['xyxy']), sample['window']['max_frames'])
            t = np.asarray(sample['boxes']['times'][:n])
            frames = np.rint(it['window']['start_frame'] + t * it['fps']).astype(int)
            with np.load(Path('data/compact') / (it['trajectory'] + '.npz')) as z:
                j = list(map(str, z['names'])).index(it['object_name'])
                corners = np.asarray(it['object']['corners'])
                points = np.stack([z['position'][f,j] + corners @ quat_to_matrix(z['quaternion_xyzw'][f,j]).T for f in frames])
            camera = Camera.from_dict(it['camera'], it['image_size'])
            uv = camera.project(points)
            oracle = np.hstack([uv.min(axis=1), uv.max(axis=1)])
            observed = np.asarray(sample['boxes']['xyxy'][:n])
            visible = sample['window'].get('edges_visible', {})
            seen = np.array([visible.get(str(f), [True]*4) for f in sample['boxes']['frames'][:n]])
            limits = np.array([camera.width, camera.height, camera.width, camera.height])
            seen &= (oracle > 1) & (oracle < limits-1) & (observed > 1) & (observed < limits-1)
            edges = {name: edge_stats(t[seen[:, k]], (observed-oracle)[seen[:, k], k])
                     for k, name in enumerate(['left','top','right','bottom'])}
            result.append({'id': rec['id'], 'fps': fps, 'event': it['event'], 'camera_id': it['camera_id'],
                           'reference_frames': frames.tolist(), 'edges': edges})
    out = {'scope': '20 error-selected development cases; oracle used only to diagnose observation errors, not to correct predictions',
           'interpretation': 'curvature in edge error can confound gravity when initial velocity is unknown; no error-based frame removal',
           'records': result}
    Path('docs/phyediting/edge_diagnosis.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps([{'fps': r['fps'], 'id':r['id'], 'max_curvature_px':max(e.get('curvature_excursion_px',0) for e in r['edges'].values())} for r in result], indent=2))

if __name__ == '__main__':
    main()
