"""What one site holds for a benchmark, as JSON. Run by the driver as `python - ARGS < survey.py` (read-only).

items:     per run/model directory: manifest rows, finished videos, their newest modification times, worker events
eval:      per model/stage (and the table): start/end events of the evaluation jobs
published: per run/model: whether the evaluation manifest was written by physedit or predates it
"""
import glob
import json
import os
import sys


def events(path):
    try:
        return [l.split() for l in open(path, errors="replace") if l.startswith(("start ", "end "))][-12:]
    except OSError:
        return []


args = json.loads(sys.argv[1])
out = {"items": {}, "eval": {}, "published": {}}
for d in glob.glob(args["dir"] + "/*/*/"):
    run, model = d.rstrip("/").split("/")[-2:]
    if run in ("eval", "input"):
        continue
    manifest = d + "manifest.jsonl"
    times = sorted((os.path.getmtime(p) for p in glob.glob(d + "generated/*/resized.mp4") if os.path.getsize(p)), reverse=True)
    out["items"][run + "/" + model] = dict(
        rows=sum(1 for _ in open(manifest)) if os.path.exists(manifest) else None, done=len(times), recent=times[:21],
        events={w: events("%slogs/%s.events" % (d, w)) for w in ("w0", "w1")})
for f in glob.glob(args["dir"] + "/eval/*.events") + glob.glob(args["dir"] + "/eval/*/*.events"):
    out["eval"][os.path.relpath(f, args["dir"] + "/eval")[:-len(".events")]] = events(f)
for key, path in args.get("publish", {}).items():
    try:
        with open(path) as f:
            out["published"][key] = "physedit" if "physedit" in json.loads(f.readline()) else "legacy"
    except (OSError, ValueError):
        pass
print(json.dumps(out))
