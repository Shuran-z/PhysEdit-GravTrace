"""Generation and evaluation of any set of models on a benchmark, driven from the hub.

One pass: survey the hosts; deploy the runners, copy the benchmark inputs and build manifests where needed; keep up
to two workers per model and run generating on free GPUs (one walks the manifest forwards, one backwards; finished
rows are skipped); copy finished videos to the benchmark's home host; publish finished runs and start every
evaluation stage whose inputs are ready; rebuild the table once all stages are done. All state lives on the hosts,
so a pass can be repeated at any time, and `--watch` repeats it until nothing is left to do.
"""
import collections
import concurrent.futures
import json
import os
import time

from . import config
from .build import FORMAT
from .fleet import Host, copy, mirror

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = Host("hub")
HUB_DIR = os.path.expanduser("~/.physedit")          # lock files and the logs of stages that run on the hub
VIDEOS = "-path './generated/*/resized.mp4' -size +0"  # find expression for a run's finished videos
RESTARTS, RESTART_WINDOW = 3, 3 * 3600                # a worker started this often recently is left for a human
FAILED = object()


def last(events, kind):
    return next((e for e in reversed(events) if e[0] == kind), None)


class Driver:
    def __init__(self, bench, models, runs=None, tag="", limit=None, hosts=None, dry=False, retry=False, log=print):
        self.b = config.BENCHMARKS[bench]
        self.base, self.bench = bench, bench + ("-" + tag if tag else "")
        self.models, self.runs = models, runs or list(self.b["runs"])
        self.limit, self.dry, self.retry, self.log = limit, dry, retry, log
        self.evaluating = not tag                       # a tagged (trial) run is generation only
        self.home = self.b["home"]
        self.hosts = {n: Host(n, **h) for n, h in config.HOSTS.items()}    # all are probed: jobs run anywhere
        self.allowed = set(hosts or self.hosts)                              # new work goes only here
        self.deployed, self.staged = set(), set()

    # ---------------------------------------------------------------- paths
    def root(self, site):
        return config.SITES[site]["root"]

    def rundir(self, site, run, model):
        return "%s/%s/%s/%s" % (self.root(site), self.bench, run, model)

    def tools(self, site):
        return self.root(site) + "/tools"

    def inputs(self, site):
        return "%s/%s/input" % (self.root(site), self.base)

    def readable(self, it):
        """The directory from which the home host reads this run's videos."""
        site = it["site"]
        shared = site == self.home or config.SITES[site].get("shared_with") == self.home
        return self.rundir(site if shared else self.home, it["run"], it["model"])

    # ---------------------------------------------------------------- state
    def survey(self):
        """Probe every host (GPU memory, our jobs) and read every involved site, all in parallel."""
        self.notes, self.taken = {}, collections.defaultdict(set)
        sites = {self.home} | {s for m in self.models for s in config.MODELS[m]["cmd"]}
        with concurrent.futures.ThreadPoolExecutor(16) as ex:
            probes = dict(zip(self.hosts, ex.map(lambda h: h.probe(), self.hosts.values())))
            self.up = {n: p for n, p in probes.items() if p}
            self.hub_jobs = (HUB.probe() or ({}, {}))[1]
            reads = {s: ex.submit(self.read_site, s) for s in sites}
            self.sites = {s: f.result() for s, f in reads.items()}
        return self.items()

    def site_hosts(self, site):
        return [n for n in self.up if self.hosts[n].site == site]

    def placeable(self, site):
        return [n for n in self.site_hosts(site) if n in self.allowed]

    def host(self, site):
        hosts = self.site_hosts(site)
        return self.hosts[hosts[0]] if hosts else None

    def read_site(self, site):
        h = self.host(site)
        if h is None:
            return None
        args = {"dir": "%s/%s" % (self.root(site), self.bench)}
        if site == self.home:
            args["publish"] = {"%s/%s" % (r, m): p.format(model=m) for r, p in self.b.get("publish", {}).items()
                               for m in self.models}
        survey = open(os.path.join(HERE, "survey.py")).read()
        try:
            return json.loads(h.sh("%s - '%s' <<'PHYSEDIT_EOF'\n%sPHYSEDIT_EOF\n"
                                   % (config.SITES[site]["vars"]["python"], json.dumps(args), survey)))
        except (RuntimeError, ValueError):
            return None

    def site_items(self, site):
        return ((self.sites.get(site) or {}).get("items")) or {}

    def items(self):
        """One item per model and run: where it runs, its rows, finished videos and live workers."""
        out = []
        for m in self.models:
            for run in self.runs:
                key, it = "%s/%s" % (run, m), dict(model=m, run=run, site=None, rows=None, done=0, recent=[], events={})
                for site in config.MODELS[m]["cmd"]:
                    info = self.site_items(site).get(key)
                    if info and info["rows"]:
                        it.update(info, site=site)
                        break
                it["live"] = {job.rsplit(".", 1)[1]: (h, g) for h, (_, jobs) in self.up.items() for job, g in jobs.items()
                              if job.startswith("%s/%s/%s." % (self.bench, m, run))}
                synced = it["site"] and self.readable(it) == self.rundir(self.home, run, m) and it["site"] != self.home
                it["at_home"] = (self.site_items(self.home).get(key) or {}).get("done", 0) if synced else it["done"]
                out.append(it)
        return out

    def free(self, host):
        """Idle GPUs of a reachable host that run none of our jobs and were not handed out in this pass."""
        used, jobs = self.up[host]
        ours = {int(g) for gs in jobs.values() for g in gs.split(",") if g} | self.taken[host]
        return [g for g, mib in sorted(used.items()) if mib <= self.hosts[host].busy_mib and g not in ours]

    def act(self, what, fn, *args):
        self.log(("[dry-run] " if self.dry else "") + what)
        if not self.dry:
            return fn(*args)

    def guard(self, key, fn, *args):
        """Run one item's step; a failure becomes a note instead of stopping the other models."""
        try:
            return fn(*args)
        except Exception as e:
            self.notes[key] = "error: " + (str(e).strip().splitlines() or [repr(e)])[-1][:300]
            return FAILED

    # ---------------------------------------------------------------- one pass
    def step(self):
        items = self.survey()
        for it in items:
            self.guard((it["model"], it["run"]), self.prepare, it)
        self.generate(items)
        for it in items:
            self.guard((it["model"], it["run"], "copy"), self.sync, it)
        if self.evaluating:
            self.evaluate(items)
        return items

    def spec(self, it, site):
        """What build.py needs to write this run's manifests on `site` (JSON-normalised, so it compares with the stored one)."""
        m = config.MODELS[it["model"]]
        mode, home = m.get("mode", "i2v"), site == self.home
        spec = dict(dir=self.rundir(site, it["run"], it["model"]), run=it["run"], model=it["model"], name=m["name"],
                    frames=m["frames"], fps=m["fps"], canvas=m.get("canvas"), fields=m.get("fields", {}),
                    letterbox=m.get("letterbox"), prompts_json=m.get("prompts_json"), split=m.get("split"),
                    split_orientation=m.get("split_orientation"), limit=self.limit, format=FORMAT,
                    template=self.b["templates"][mode][it["run"]] if home else
                    "%s/templates/%s_%s.jsonl" % (self.inputs(site), mode, it["run"]),
                    inputs={n: p if home else "%s/%s" % (self.inputs(site), n) for n, p in self.b["inputs"].items()},
                    remap=[] if home else [[p, "%s/%s" % (self.inputs(site), n)] for n, p in self.b["inputs"].items()])
        return json.loads(json.dumps(spec))

    def prepare(self, it):
        """Pin an unstarted run to the candidate site with most free GPUs; deploy, stage inputs, build its manifest.
        A started run whose model settings changed is rebuilt while it has no videos, and reported otherwise."""
        if it["site"]:
            site, spec = it["site"], self.spec(it, it["site"])
            if it.get("spec") == spec:
                return
            if it["done"]:
                self.notes[(it["model"], it["run"])] = ("model settings changed after %d videos were made; remove %s to "
                                                         "start this run again" % (it["done"], spec["dir"]))
                return
        else:
            cands = [s for s in config.MODELS[it["model"]]["cmd"] if self.placeable(s)]
            if not cands:
                self.notes[(it["model"], it["run"])] = "no usable host on %s" % "/".join(config.MODELS[it["model"]]["cmd"])
                return
            site = max(cands, key=lambda s: sum(len(self.free(h)) for h in self.placeable(s)))
            spec = self.spec(it, site)
        self.deploy(site)
        self.stage(site)
        h = self.host(site)

        def build():
            h.write(spec["dir"] + "/spec.json", json.dumps(spec, indent=1))
            out = h.sh("%s %s/build.py %s/spec.json" % (config.SITES[site]["vars"]["python"], self.tools(site), spec["dir"]))
            return json.loads(out.strip().splitlines()[-1])["rows"]
        rows = self.act("build %s/%s manifest on %s" % (it["run"], it["model"], site), build)
        it.update(site=site, rows=rows if rows else self.limit or 1, planned=self.dry)

    def deploy(self, site):
        """Copy the runners and the manifest builder to the site (once per driver process)."""
        if site in self.deployed:
            return
        h = self.host(site)
        runners = sorted(f for f in os.listdir(os.path.join(HERE, "runners")) if f.endswith(".py"))
        self.act("deploy runners to %s:%s" % (site, self.tools(site)), lambda: (
            copy(HUB, os.path.join(HERE, "runners"), h, self.tools(site), runners),
            copy(HUB, HERE, h, self.tools(site), ["build.py"])))
        self.deployed.add(site)

    def stage(self, site):
        """Copy the benchmark's inputs and templates from the home host to another site (once)."""
        if site == self.home or site in self.staged:
            return
        h, home, dst = self.host(site), self.host(self.home), self.inputs(site)
        if h.sh("test -f %s/.staged && echo yes || true" % dst).strip() != "yes":
            def run():       # resumable: only files missing (or of another size) on the site are sent
                for name, path in self.b["inputs"].items():
                    mirror(home, path, h, "%s/%s" % (dst, name))
                for mode, runs in self.b["templates"].items():
                    for run, path in runs.items():
                        h.write("%s/templates/%s_%s.jsonl" % (dst, mode, run), home.sh("cat " + path))
                h.sh("date > %s/.staged" % dst)
            self.act("stage the %s inputs on %s:%s" % (self.base, site, dst), run)
        self.staged.add(site)

    def generate(self, items):
        """Keep workers on every unfinished run: first worker for every run, then second workers."""
        now = time.time()
        for k in (0, 1):
            for it in items:
                m, w = config.MODELS[it["model"]], "w%d" % k
                if not it["site"] or it["rows"] is None or it["done"] >= it["rows"] or k >= m.get("workers", 2) \
                        or w in it["live"]:
                    continue
                ev = it["events"].get(w, [])
                if not self.retry and sum(1 for e in ev if e[0] == "start" and now - float(e[1]) < RESTART_WINDOW) >= RESTARTS:
                    self.notes[(it["model"], it["run"], w)] = "started %d times in %d h; see %s/logs/%s.log" % (
                        RESTARTS, RESTART_WINDOW // 3600, self.rundir(it["site"], it["run"], it["model"]), w)
                    continue
                end, start = last(ev, "end"), last(ev, "start")
                if it["live"] and end and end[2] == "rc=0" and float(end[1]) >= float(start[1]):
                    continue                    # it walked the whole manifest; the other worker is on the last rows
                self.guard((it["model"], it["run"], w), self.launch, it, k)

    def launch(self, it, k):
        site, m = it["site"], config.MODELS[it["model"]]
        n = m.get("gpus", 1)
        host = next((h for h in self.placeable(site) if len(self.free(h)) >= n), None)
        if host is None:
            self.notes[(it["model"], it["run"], "w%d" % k)] = "waiting for %d free GPU%s on %s" % (n, "s" * (n > 1), site)
            return
        gpus = self.free(host)[:n]
        gpu, d = ",".join(map(str, gpus)), self.rundir(site, it["run"], it["model"])
        job = "%s/%s/%s.w%d" % (self.bench, it["model"], it["run"], k)
        values = dict(config.SITES[site]["vars"], **self.hosts[host].vars)
        values.update(manifest="%s/manifest_w%d.jsonl" % (d, k), dir=d,
                      tools=self.tools(site), gpu=gpu, run=it["run"], worker="w%d" % k, job=job.replace("/", "."))
        cmd = "export CUDA_VISIBLE_DEVICES=%s %s; %s" % (gpu, config.SITES[site]["env"], m["cmd"][site].format(**values))
        self.taken[host] |= set(gpus)
        self.act("start %s on %s GPU %s" % (job, host, gpu), self.hosts[host].spawn, job, gpu, cmd,
                 "%s/logs/w%d.log" % (d, k), "%s/logs/w%d.events" % (d, k))
        it["live"]["w%d" % k] = (host, gpu)

    def sync(self, it):
        """Copy finished videos of a site the home host cannot read to the home host."""
        home, src = self.host(self.home), it["site"] and self.host(it["site"])
        if not src or not home or it["done"] <= it["at_home"]:
            return
        self.act("copy the new videos of %s/%s from %s to %s (%d there, %d here)" % (
            it["run"], it["model"], it["site"], self.home, it["done"], it["at_home"]),
            mirror, src, self.rundir(it["site"], it["run"], it["model"]), home, self.readable(it), VIDEOS)

    # ---------------------------------------------------------------- evaluation
    def running(self):
        return {j for jobs in [self.hub_jobs] + [j for _, j in self.up.values()] for j in jobs}

    def stage_state(self, model, stage):
        """done / failed / running, or None (never run, or interrupted before an exit code was written)."""
        if "%s/%s/%s" % (self.bench, model, stage) in self.running():
            return "running"
        key = "%s/%s" % (model, stage)
        events = ((self.sites.get(self.home) or {}).get("eval") or {}).get(key) or self.hub_events(key)
        end, start = last(events, "end"), last(events, "start")
        if end and (not start or float(end[1]) >= float(start[1])):
            return "done" if end[2] == "rc=0" else "failed"
        return None

    def hub_events(self, key):
        try:
            return [l.split() for l in open("%s/%s/eval/%s.events" % (HUB_DIR, self.bench, key))]
        except OSError:
            return []

    def ours(self, published):
        return [m for m in self.models if any(published.get("%s/%s" % (r, m)) in ("physedit", "dry-run") for r in self.runs)]

    def evaluate(self, items):
        home = self.host(self.home)
        if home is None or not self.sites.get(self.home):
            return
        published = self.sites[self.home]["published"]
        for it in items:                                   # publish finished runs for the evaluation scripts
            key = "%s/%s" % (it["run"], it["model"])
            if published.get(key) == "legacy":
                self.notes[(it["model"], it["run"], "eval")] = "evaluated before physedit; left untouched"
            elif it["rows"] and it["at_home"] >= it["rows"] and key not in published and it["run"] in self.b["publish"]:
                if self.guard((it["model"], it["run"], "publish"), self.publish, it) is not FAILED:
                    published[key] = "dry-run" if self.dry else "physedit"
        ours = self.ours(published)
        legacy = self.b.get("legacy", [])
        vp_models = ",".join(legacy + [m for m in ours if published.get("x5/" + m) == "physedit" and m not in legacy])
        for m in ours:
            for st in self.b["evaluate"]:
                state, after = self.stage_state(m, st["stage"]), st["after"]
                ready = published.get("%s/%s" % (after, m)) in ("physedit", "dry-run") if after in self.b["runs"] \
                    else self.stage_state(m, after) == "done"
                if state in ("done", "running") or (state == "failed" and not self.retry) or not ready:
                    continue
                if st.get("exclusive") and any(j.startswith(self.bench + "/") and j.endswith("/" + st["stage"])
                                               for j in self.running()):
                    continue
                self.guard((m, st["stage"]), self.run_stage, m, st, vp_models)
        if ours and all(self.stage_state(m, st["stage"]) == "done" for m in ours for st in self.b["evaluate"]) \
                and self.stage_state("all", "table") not in ("done", "running"):
            self.guard(("table",), self.run_stage, "all", dict(stage="table", cmd=self.b["table"]), vp_models)

    def run_stage(self, model, st, vp_models):
        on_hub, n = st.get("where") == "hub", st.get("gpus", 0)
        home = self.host(self.home)
        gpus = self.free(home.name)[:n] if n and home.name in self.allowed else []
        if len(gpus) < n:
            self.notes[(model, st["stage"])] = "waiting for %d free GPUs on %s" % (n, self.home)
            return
        self.taken[home.name] |= set(gpus)
        gpu = ",".join(map(str, gpus))
        values = dict(config.SITES[self.home]["vars"], model=model, gpu=gpu, vp_models=vp_models)
        cmd = ("" if on_hub else "export %s; " % config.SITES[self.home]["env"]) + st["cmd"].format(**values)
        d = "%s/%s/eval/%s" % (HUB_DIR if on_hub else self.root(self.home), self.bench, model)
        name = "%s/%s/%s" % (self.bench, model, st["stage"])
        self.act("start %s on %s%s" % (name, "hub" if on_hub else self.home, " GPU " + gpu if gpu else ""),
                 (HUB if on_hub else home).spawn, name, gpu, cmd, "%s/%s.log" % (d, st["stage"]),
                 "%s/%s.events" % (d, st["stage"]))

    def publish(self, it):
        """Write the run's manifest where the benchmark's evaluation scripts read it, with home-readable video paths."""
        src, home = self.host(it["site"]), self.host(self.home)
        d, rd = self.rundir(it["site"], it["run"], it["model"]), self.readable(it)
        path = self.b["publish"][it["run"]].format(model=it["model"])

        def run():
            if rd != d:      # a copy interrupted earlier would have left a short file: compare sizes once more
                mirror(src, d, home, rd, VIDEOS)
            rows = [json.loads(l) for l in src.sh("cat %s/manifest.jsonl" % d).splitlines() if l.strip()]
            for r in rows:
                r["generated_video_resized_to_source"] = "%s/generated/%s/resized.mp4" % (rd, r["sample_id"])
                r["physedit"] = dict(bench=self.bench, site=it["site"], run=it["run"])
            home.write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        self.act("publish %s/%s for evaluation: %s" % (it["run"], it["model"], path), run)

    # ---------------------------------------------------------------- report
    def finished(self, items):
        if not all(it["rows"] and not it.get("planned") and it["at_home"] >= it["rows"] for it in items):
            return False
        if not self.evaluating:
            return True
        ours = self.ours(self.sites[self.home]["published"]) if self.sites.get(self.home) else []
        return all(self.stage_state(m, st["stage"]) == "done" for m in ours for st in self.b["evaluate"]) and \
            (not ours or self.stage_state("all", "table") == "done")

    def report(self, items):
        now = time.time()
        lines = ["%s  %s" % (self.bench, time.strftime("%Y-%m-%d %H:%M %Z")),
                 "%-26s %-3s %-9s %9s %6s  %-11s %s" % ("model", "run", "site", "done", "s/row", "ETA", "workers")]
        for it in items:
            rows, done, t = it["rows"], it["done"], it["recent"]
            rate = (t[0] - t[-1]) / (len(t) - 1) if len(t) > 1 and it["live"] else None
            if rate is None and it["live"]:
                rate = config.MODELS[it["model"]]["sec_per_row"] / len(it["live"])
            if it.get("planned"):
                eta = "planned"
            elif rows and done >= rows:
                eta = "done" if it["at_home"] >= rows else "copying"
            elif rate:
                eta = time.strftime("%m-%d %H:%M", time.localtime(now + (rows - done) * rate))
            else:
                eta = "waiting" if it["site"] else "not started"
            workers = " ".join("%s=%s:%s" % (w, h, g) for w, (h, g) in sorted(it["live"].items()))
            lines.append("%-26s %-3s %-9s %9s %6s  %-11s %s" % (
                it["model"], it["run"], it["site"] or "-", "%d/%s" % (done, "?" if it.get("planned") else rows or "?"),
                "%.0f" % rate if rate else "", eta, workers))
        if self.evaluating and self.sites.get(self.home):
            names = [st["stage"] for st in self.b["evaluate"]]
            lines.append("\n%-26s " % "evaluation" + " ".join("%-10s" % n for n in names))
            published = self.sites[self.home]["published"]
            for m in self.models:
                legacy = all(published.get("%s/%s" % (r, m)) == "legacy" for r in self.runs)
                lines.append("%-26s " % m + ("evaluated before physedit" if legacy else
                                            " ".join("%-10s" % (self.stage_state(m, n) or "-") for n in names)))
            lines.append("table: %s" % (self.stage_state("all", "table") or "-"))
        lines += ["note  %s: %s" % ("/".join(k), v) for k, v in sorted(self.notes.items())]
        down = sorted(set(self.hosts) - set(self.up))      # every host is probed, whatever --hosts says
        if down:
            lines.append("unreachable: " + " ".join(down))
        return "\n".join(lines)
