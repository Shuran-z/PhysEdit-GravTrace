"""A generated video inherits its ground-truth scene; only masks, fps and first-frame time change."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generated_manifest.py"


GT = {"id": "scene", "scenario": "freefall", "fps": 30.0, "motion": {"t0": [0.05, 0.07]},
      "window": {"start_frame": 14, "max_frames": 5}, "masks": {"dir": "gt"}, "truth": {"gravity": 9.81}}


def convert(tmp_path, video):
    (tmp_path / "gt.jsonl").write_text(json.dumps(GT) + "\n")
    (tmp_path / "videos.jsonl").write_text(json.dumps(video) + "\n")
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "gt.jsonl"), str(tmp_path / "videos.jsonl"),
                    str(tmp_path / "out.jsonl")], check=True)
    return json.loads((tmp_path / "out.jsonl").read_text())


def test_generated_video_inherits_scene(tmp_path):
    out = convert(tmp_path, {"id": "model_scene", "gt_id": "scene", "fps": 16.0, "start_s": 0.1,
                             "masks": {"dir": "gen", "pattern": "mask_*.png", "label": None}})
    assert out["id"] == "model_scene" and out["fps"] == 16.0 and out["masks"]["dir"] == "gen"
    assert out["truth"] == GT["truth"] and out["window"] == {"max_frames": 5}
    assert all(abs(a - b) < 1e-12 for a, b in zip(out["motion"]["t0"], [0.15, 0.17]))


def test_clip_starting_early_is_fitted_from_the_same_instant(tmp_path):
    frames = list(range(10))
    out = convert(tmp_path, {"id": "model_scene", "gt_id": "scene", "fps": 16.0, "start_s": -0.3,
                             "boxes": {"frames": frames, "xyxy": [[0, 0, 1, 1]] * 10, "times": [f / 16 for f in frames]}})
    assert "masks" not in out and out["window"] == {"max_frames": 5, "start_frame": 5}  # 5/16 s is the first at or after 0.3 s
    assert all(abs(a - b) < 1e-12 for a, b in zip(out["motion"]["t0"], [0.0625, 0.0825]))
