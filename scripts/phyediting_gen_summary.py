"""Generation-test summary for the report (reference-free launch scoring, `score --v0 agnostic`; see phyediting_gen_nulls.py).

    python gen_summary.py RUN_DIR:MODEL [RUN_DIR:MODEL ...] > gen.json

RUN_DIR holds rows_<m>.jsonl and pred_<m>_agnostic.jsonl. Run from the repo root.
"""
import json, sys
import numpy as np
from scipy.stats import spearmanr

specs = [a.rsplit(":", 1) for a in sys.argv[1:]]
items = {json.loads(l)["id"]: json.loads(l) for l in open("runs/benchmark_gravity_v1/items.jsonl")}
LABEL = {"ltx_i2v": "LTX-Video-2B", "wan_ti2v5b": "Wan2.2-TI2V-5B", "cogvideox_i2v": "CogVideoX1.5-5B",
         "cosmos3_nano": "Cosmos3-Nano", "wan22_i2v_a14b": "Wan2.2-I2V-A14B", "physalign_wan22_i2v_a14b": "PhysAlign（Wan2.2-A14B）"}
TARGETS = (1.62, 3.71, 9.81, 15.0, 24.0)
VARIANT = "agnostic"
LOW = 0.5  # m/s^2: a third of the smallest target; no ground-truth video reads below 1.3


def scan_hits(path, wanted):
    """Most lenient reading: does some window anywhere in the clip read within 20 % of the target? Returns the
    number of videos that hit their own target, and the hit rate for the four other targets as a target-substitution diagnostic, not a significance test."""
    try:
        scans = {}
        for line in open(path):
            rec = json.loads(line)
            scans[rec["item_id"]] = rec
    except FileNotFoundError:
        return None, None
    own, other = 0, []
    for item_id, target in wanted:
        vals = [g for _, g, st in scans.get(item_id, {}).get("scan", []) if st == "ok" and g is not None]
        hit = lambda t: bool(vals) and min(abs(g / t - 1) for g in vals) <= 0.2
        own += hit(target)
        other += [hit(t) for t in TARGETS if abs(t - target) > 1e-6]
    return own, round(100 * float(np.mean(other)), 0) if other else None


def summarise(preds, wanted):
    """preds: item_id -> prediction; wanted: (item_id, target) pairs of the test set."""
    pts = []
    for item_id, target in wanted:
        p = preds.get(item_id)
        if p and p.get("status") in ("ok", "at_bound"):
            it = items[item_id]
            pts.append(dict(target=target, inverted=round(p["gravity"], 3), event=it["event"], camera=it["camera_id"],
                            at_bound=p["status"] == "at_bound"))
    errs = np.array([abs(p["inverted"] / p["target"] - 1) for p in pts])
    moving = [p for p in pts if p["inverted"] > LOW]
    rho = spearmanr([p["target"] for p in moving], [p["inverted"] for p in moving]).correlation if len(moving) > 4 else None
    per = []
    for g in TARGETS:
        v = [p["inverted"] for p in pts if abs(p["target"] - g) < 1e-6]
        if v:
            per.append(dict(target=g, n=len(v), median=round(float(np.median(v)), 3)))
    return dict(scored=len(pts), low=len(pts) - len(moving), within20=int((errs <= 0.2).sum()),
                median_err=round(float(np.median(errs) * 100), 1) if errs.size else None,
                rho=None if rho is None or np.isnan(rho) else round(float(rho), 2), points=pts, per_target=per)


out = {"models": [], "refs": []}
wanted = None
for d, m in specs:
    rows = [json.loads(l) for l in open(f"{d}/rows_{m}.jsonl")]
    try:
        by_sample = {json.loads(l)["id"]: json.loads(l) for l in open(f"{d}/pred_{m}_{VARIANT}.jsonl")}
    except FileNotFoundError:
        continue
    preds = {r["item_id"]: by_sample[r["sample_id"]] for r in rows if r["sample_id"] in by_sample}
    wanted = [(r["item_id"], r["gravity_truth"]) for r in rows]
    out["models"].append(dict(key=m, label=LABEL.get(m, m), generated=len(rows), **dict(zip(("scan20", "scan_other"), scan_hits(f"{d}/scan_{m}.jsonl", wanted))),
                              **summarise(preds, wanted)))

if wanted:  # the two null generators on the same items
    for k, label in (("static", "参照：物体始终不动"), ("earth", "参照：始终按地球重力")):
        preds = {json.loads(l)["item_id"]: json.loads(l) for l in open(f"runs/gen_nulls/pred_{k}_{VARIANT}.jsonl")}
        s = summarise(preds, wanted)
        s.pop("points"); s.pop("per_target")
        out["refs"].append(dict(key=k, label=label, generated=len(wanted), **dict(zip(("scan20", "scan_other"), scan_hits(f"runs/gen_nulls_full/scan_{k}.jsonl", wanted))), **s))


def floor(path):
    rows = [json.loads(l) for l in open(path)]
    e = np.array([abs(r["gravity"] / r["gravity_target"] - 1) for r in rows if r.get("status") in ("ok", "at_bound")]) * 100
    return dict(n=int(e.size), mean=round(float(e.mean()), 2), max=round(float(e.max()), 1))


def leak(path):
    rows = [json.loads(l) for l in open(path)]
    g = {}
    for r in rows:
        if r.get("status") in ("ok", "at_bound"):
            g.setdefault(r["gravity_target"], []).append(r["gravity"])
    return {str(t): round(float(np.median(v)), 2) for t, v in sorted(g.items())}


out["floor"] = floor(f"runs/genfloor_pred_30_{VARIANT}.jsonl")
out["floor15"] = floor(f"runs/genfloor_pred_15_{VARIANT}.jsonl")
out["leak"] = {f"{k}_{v}": leak(f"runs/gen_nulls/pred_{k}_{v}.jsonl") for k in ("static", "earth") for v in ("declared", "free", "agnostic")}
print(json.dumps(out, ensure_ascii=False))
