import json
import os

import pytest

from physedit.build import build


def template(tmp, benchmarks=("pisa", "newton"), sources=2, size=(1280, 720)):
    """Template rows as the benchmark ships them: one per source and prompt granularity, paths under /home/in."""
    rows = []
    for b in benchmarks:
        for i in range(sources):
            sid = "%s_%d" % (b, i)
            for g in ("coarse", "fine"):
                rows.append(dict(index=len(rows), sample_id="ltx_i2v_%s_%s" % (sid, g), source_sample_id=sid, benchmark=b,
                                 prompt_granularity=g, source_width=size[0], source_height=size[1], seed=20260920,
                                 condition_image="/home/in/cond/%s.png" % sid, prompt_file="/home/in/prompts/%s/%s/prompt.txt" % (sid, g),
                                 negative_prompt_file="/home/in/prompts/%s/%s/negative_prompt.txt" % (sid, g),
                                 gen_width=999, model_name="LTX"))
    path = tmp / "template.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:   # the staged copies on this site
        for k in ("condition_image", "prompt_file", "negative_prompt_file"):
            f = tmp / "in" / os.path.relpath(r[k], "/home/in")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("a ball falls")
    return path


def spec(tmp, **kw):
    s = dict(dir=str(tmp / "run"), run="x5", model="m", name="Model", frames=49, fps=15,
             inputs={"prefix5": str(tmp / "in" / "prefix5")}, remap=[["/home/in", str(tmp / "in")]])
    s.update(kw)
    if "template" not in s:
        s["template"] = str(template(tmp))
    return s


def read(path):
    return [json.loads(l) for l in open(path)]


def test_rows_paths_and_order(tmp_path):
    assert build(spec(tmp_path, canvas={"landscape": [832, 480], "portrait": [480, 832], "square": [640, 640]})) == 8
    rows = read(tmp_path / "run" / "manifest.jsonl")
    # round-robin over benchmarks, coarse before fine of each source
    assert [r["source_sample_id"] for r in rows] == ["pisa_0", "pisa_0", "newton_0", "newton_0", "pisa_1", "pisa_1",
                                                     "newton_1", "newton_1"]
    r = rows[0]
    assert r["sample_id"] == "m_pisa_0_coarse" and r["model_name"] == "Model" and r["num_frames"] == 49
    assert r["generated_video_resized_to_source"] == str(tmp_path / "run" / "generated" / "m_pisa_0_coarse" / "resized.mp4")
    assert r["condition_image"] == str(tmp_path / "in" / "cond" / "pisa_0.png")      # remapped to the staged copy
    assert (r["gen_width"], r["gen_height"]) == (832, 480) and r["orientation"] == "landscape"
    assert read(tmp_path / "run" / "manifest_rev.jsonl") == rows[::-1]


def test_fields_per_run_limit_and_missing_inputs(tmp_path):
    fields = {"steps": 35, "x5": {"source_video": "{inputs[prefix5]}/{source_sample_id}.mp4"}}
    with pytest.raises(SystemExit, match="source_video"):                  # prefix videos not staged yet
        build(spec(tmp_path, fields=fields))
    (tmp_path / "in" / "prefix5").mkdir()
    for sid in ("pisa_0", "newton_0"):
        (tmp_path / "in" / "prefix5" / (sid + ".mp4")).write_text("v")
    assert build(spec(tmp_path, fields=fields, limit=4)) == 4
    rows = read(tmp_path / "run" / "manifest.jsonl")
    assert rows[0]["source_video"] == str(tmp_path / "in" / "prefix5" / "pisa_0.mp4") and rows[0]["steps"] == 35
    assert "gen_width" not in rows[0]                                       # the template's canvas is not inherited
    assert "source_video" not in read_x1(tmp_path, fields)[0]              # per-run field only for x5


def read_x1(tmp_path, fields):
    build(spec(tmp_path, run="x1", dir=str(tmp_path / "x1"), fields=fields, limit=2))
    return read(tmp_path / "x1" / "manifest.jsonl")


def test_orientation_split_and_prompt_files(tmp_path):
    build(spec(tmp_path, split_orientation=True, prompts_json=True))
    run = tmp_path / "run"
    assert len(read(run / "manifest_landscape.jsonl")) == 8 and read(run / "manifest_portrait.jsonl") == []
    prompts = json.load(open(run / "prompts_rev.json"))
    assert list(prompts)[0] == "m_newton_1_fine" and prompts["m_pisa_0_coarse"]["text_prompt"] == "a ball falls"


def test_letterbox_portrait_sources(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    s = spec(tmp_path, letterbox=True, template=str(template(tmp_path, size=(480, 640))))
    for sid in ("pisa_0", "pisa_1", "newton_0", "newton_1"):
        Image.new("RGB", (480, 640), "white").save(tmp_path / "in" / "cond" / (sid + ".png"))
    build(s)
    r = read(tmp_path / "run" / "manifest.jsonl")[0]
    x0, y0, w, h = r["letterbox_box"]           # 16:9 canvas, both sides rounded up as in the September runs
    assert Image.open(r["condition_image"]).size == (1138, 641)
    assert abs(x0 * 1138 - 329) < 1e-9 and abs(w * 1138 - 480) < 1e-9 and abs(h * 641 - 640) < 1e-9
    assert r["final_resized"].endswith("/resized.mp4") and r["generated_video_resized_to_source"].endswith("unused.mp4")
