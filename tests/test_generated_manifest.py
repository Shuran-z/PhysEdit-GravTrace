"""A generated video inherits its ground-truth scene; only masks, fps and first-frame time change."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generated_manifest.py"


def test_generated_video_inherits_scene(tmp_path):
    gt = {"id": "scene", "scenario": "freefall", "fps": 30.0, "motion": {"t0": [0.05, 0.07]},
          "window": {"start_frame": 14, "max_frames": 5}, "masks": {"dir": "gt"}, "truth": {"gravity": 9.81}}
    video = {"id": "model_scene", "gt_id": "scene", "fps": 16.0, "start_s": 0.1,
             "masks": {"dir": "gen", "pattern": "mask_*.png", "label": None}}
    (tmp_path / "gt.jsonl").write_text(json.dumps(gt) + "\n")
    (tmp_path / "videos.jsonl").write_text(json.dumps(video) + "\n")
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "gt.jsonl"), str(tmp_path / "videos.jsonl"),
                    str(tmp_path / "out.jsonl")], check=True)
    out = json.loads((tmp_path / "out.jsonl").read_text())
    assert out["id"] == "model_scene" and out["fps"] == 16.0 and out["masks"]["dir"] == "gen"
    assert out["truth"] == gt["truth"] and out["window"] == {"max_frames": 5}
    assert all(abs(a - b) < 1e-12 for a, b in zip(out["motion"]["t0"], [0.15, 0.17]))
