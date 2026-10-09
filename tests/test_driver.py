"""The driver end to end on a fake fleet on this machine: two sites in a temporary directory (the second one not
readable from home, so its videos are copied), a runner that crashes once, and evaluation stages that log to a file."""
import json
import os
import sys
import time

import pytest

from physedit import config, driver, fleet

RUNNER = """import json, os, sys
manifest, d = sys.argv[1:3]
if "/x5/m1" in d and not os.path.exists(d + "/crashed"):        # first start fails: the driver must start it again
    open(d + "/crashed", "w").close(); sys.exit(1)
for line in open(manifest):
    p = json.loads(line)["generated_video_resized_to_source"]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        open(p, "w").write("video")
"""


@pytest.fixture
def fleet_setup(tmp_path, monkeypatch):
    for sid in ("s0", "s1"):
        (tmp_path / "in" / "cond").mkdir(parents=True, exist_ok=True)
        (tmp_path / "in" / "cond" / (sid + ".png")).write_text("png")
        for g in ("coarse", "fine"):
            p = tmp_path / "in" / "prompts" / sid / g
            p.mkdir(parents=True)
            (p / "prompt.txt").write_text("a ball falls")
            (p / "negative_prompt.txt").write_text("")
    for run in ("x5", "x1"):
        rows = [dict(source_sample_id=sid, benchmark="pisa", prompt_granularity=g, source_width=64, source_height=48,
                     seed=1, condition_image=str(tmp_path / "in" / "cond" / (sid + ".png")),
                     prompt_file=str(tmp_path / "in" / "prompts" / sid / g / "prompt.txt"),
                     negative_prompt_file=str(tmp_path / "in" / "prompts" / sid / g / "negative_prompt.txt"))
                for sid in ("s0", "s1") for g in ("coarse", "fine")]
        (tmp_path / ("t_%s.jsonl" % run)).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "runner.py").write_text(RUNNER)
    trace = tmp_path / "trace.txt"
    v = dict(python=sys.executable, runner=str(tmp_path / "runner.py"), trace=str(trace))
    monkeypatch.setattr(config, "SITES", {"a": dict(root=str(tmp_path / "a"), env="X=1", vars=v),
                                          "b": dict(root=str(tmp_path / "b"), env="X=1", vars=v)})
    monkeypatch.setattr(config, "HOSTS", {"ha": dict(site="a", gpus=[0, 1]), "hb": dict(site="b", gpus=[0])})
    cmd = "{python} {runner} {manifest} {dir}"
    monkeypatch.setattr(config, "MODELS", {"m1": dict(name="M1", frames=9, fps=8, sec_per_row=1, cmd={"a": cmd}),
                                           "m2": dict(name="M2", frames=9, fps=8, sec_per_row=1, cmd={"b": cmd})})
    log = "echo %s {model} gpu={gpu} >> {trace}"
    monkeypatch.setattr(config, "BENCHMARKS", {"t": dict(
        home="a", runs={"x5": "", "x1": ""}, legacy=[],
        templates={"i2v": {r: str(tmp_path / ("t_%s.jsonl" % r)) for r in ("x5", "x1")}},
        inputs={"cond": str(tmp_path / "in" / "cond"), "prompts": str(tmp_path / "in" / "prompts")},
        publish={r: str(tmp_path / "pub" / ("{model}_%s.jsonl" % r)) for r in ("x5", "x1")},
        evaluate=[dict(stage="vp", after="x5", gpus=1, exclusive=True, cmd=log % "vp"),
                  dict(stage="judge", after="vp", where="hub", cmd="echo judge {model} >> {trace}"),
                  dict(stage="inversion", after="x1", cmd=log % "inversion")],
        table="echo table {vp_models} >> {trace}")})
    monkeypatch.setattr(driver, "HUB_DIR", str(tmp_path / "hub"))
    spawned, probe, spawn = set(), fleet.Host.probe, fleet.Host.spawn

    def fake_spawn(self, job, gpus, cmd, log, events):        # every "host" is this machine: remember who started what
        spawned.add((self.name, job))
        return spawn(self, job, gpus, cmd, log, events)

    def fake_probe(self):                                     # idle GPUs, and only the jobs started on this host
        _, jobs = probe(self)
        return {g: 0 for g in self.gpus}, {j: g for j, g in jobs.items() if (self.name, j) in spawned}
    monkeypatch.setattr(fleet.Host, "spawn", fake_spawn)
    monkeypatch.setattr(fleet.Host, "probe", fake_probe)
    return tmp_path


def test_generate_copy_publish_evaluate(fleet_setup, monkeypatch):
    tmp = fleet_setup
    d = driver.Driver("t", ["m1", "m2"], log=lambda *a: None)
    for _ in range(80):
        items = d.step()
        if d.finished(items):
            break
        time.sleep(0.3)
    else:
        pytest.fail("not finished:\n" + d.report(d.survey()))
    for model, site in (("m1", "a"), ("m2", "b")):
        for run in ("x5", "x1"):
            rows = [json.loads(l) for l in open(tmp / "pub" / ("%s_%s.jsonl" % (model, run)))]
            assert len(rows) == 4 and all(r["physedit"]["site"] == site for r in rows)
            for r in rows:   # published paths point at the home copy, which exists
                assert r["generated_video_resized_to_source"].startswith(str(tmp / "a" / "t" / run / model))
                assert os.path.exists(r["generated_video_resized_to_source"])
    events = open(tmp / "a" / "t" / "x5" / "m1" / "logs" / "w0.events").read().split("\n")
    assert events[1].endswith("rc=1") and events[-2].endswith("rc=0")         # crashed once, was started again
    trace = open(tmp / "trace.txt").read().split("\n")
    line = lambda prefix: next(i for i, l in enumerate(trace) if l.startswith(prefix))
    for m in ("m1", "m2"):
        assert line("vp %s gpu=" % m) < line("judge %s" % m) and line("inversion %s gpu=" % m) >= 0
    assert [l for l in trace if l.startswith("table")] == ["table m1,m2"] and trace[-2].startswith("table")
    monkeypatch.setitem(config.MODELS["m1"], "frames", 10)       # settings changed after the videos were made:
    d.step()                                                     # reported, never mixed into the finished run
    assert "settings changed after 4 videos" in d.notes[("m1", "x5")]
