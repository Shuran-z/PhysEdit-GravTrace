"""The cleaned PhyEditing gravity benchmark: evaluation items, physics groups and a data-quality report.

An item is one (trajectory, camera) video with the window that best constrains gravity: the object is in
view and shows >= 2 unhidden box edges in >= 4 frames, GravTrace's sensitivity on the true boxes is >= --sens px
(the image changes that much between g/2 and 2g), and SAM2 follows the object in the ground-truth video
(no visible edge strays more than --dev px from the projected true box, constant offset removed).
Trajectories that differ only by background or are deterministic repeats share a physics group.

    python scripts/phyediting_benchmark.py OUT_DIR --root data/phyediting --compact data/compact \
        --oracle runs/all_oracle_cam0{1,2,3}.jsonl --oracle-pred PRED.jsonl --observed SAM2.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phyediting_report import tracking, visible_sag  # noqa: E402


def physics_group(header: dict, z) -> str:
    """Event, gravity and the moving objects' displacements (mm, every 24 frames): equal for background-only variants."""
    names = [str(n) for n in z["names"]]
    movable = sorted((i for i, o in enumerate(header["objects"]) if not o.get("fixed")), key=lambda i: names[i])
    pos = z["position"][:, movable]
    ks = [k for k in range(24, len(pos), 24)]
    rel = np.round(np.nan_to_num(pos[ks] - pos[0]) * 1000).astype(np.int64)
    key = f"{header['event']}|{header['gravity']}|" + "|".join(names[i] for i in movable)
    return hashlib.md5(key.encode() + rel.tobytes()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--root", type=Path, default=Path("data/phyediting"))
    ap.add_argument("--compact", type=Path, default=Path("data/compact"))
    ap.add_argument("--oracle", nargs="+", required=True)
    ap.add_argument("--oracle-pred", required=True)
    ap.add_argument("--observed", required=True)
    ap.add_argument("--sens", type=float, default=15.0)
    ap.add_argument("--dev", type=float, default=4.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    headers, groups = {}, {}
    for path in sorted(args.compact.glob("*.json")):
        h = json.loads(path.read_text())
        headers[h["sample_id"]] = h
        with np.load(path.with_suffix(".npz")) as z:
            groups[h["sample_id"]] = physics_group(h, z)

    sens = {}
    for line in open(args.oracle_pred):
        p = json.loads(line)
        sens[p["id"]] = p.get("sensitivity_px", 0.0) if p.get("status") == "ok" else 0.0
    observed = {}
    for line in open(args.observed):
        s = json.loads(line)
        observed[s["id"]] = s
    reasons, best = Counter(), {}
    for path in args.oracle:
        for line in open(path):
            o = json.loads(line)
            wid = o["id"]
            if len(o["boxes"]["frames"]) < 4:
                reasons["out of view"] += 1
                continue
            if visible_sag(o)[1] < 4:
                reasons["occluded"] += 1
                continue
            if sens.get(wid, 0.0) < args.sens:
                reasons["gravity barely visible"] += 1
                continue
            if wid not in observed:
                reasons["not tracked yet"] += 1
                continue
            cover, dev = tracking(o, observed[wid])
            if cover < 0.8 or dev > args.dev:
                reasons["tracker loses the object"] += 1
                continue
            key = (o["gt_id"], o["meta"]["camera"])
            if key not in best or sens[wid] > sens[best[key][0]]:
                best[key] = (wid, o)
    items = []
    for (gt_id, camera), (wid, o) in sorted(best.items()):
        h = headers[gt_id]
        items.append({
            "id": wid, "trajectory": gt_id, "camera_id": camera, "source": h["source"], "event": h["event"],
            "background": h["background"], "gravity": h["gravity"], "physics_group": groups[gt_id],
            "video": h["videos"][camera], "fps": h["fps"], "image_size": h["resolution"], "camera": o["camera"],
            "view": h["camera"][camera],
            "object": {k: v for k, v in o["object"].items()}, "object_name": wid.split("__")[-2],
            "motion": o["motion"], "window": o["window"], "sensitivity_px": round(sens[wid], 2),
        })
    with open(args.out / "items.jsonl", "w") as f:
        f.writelines(json.dumps(item) + "\n" for item in items)

    # coverage: items and distinct physics per event x gravity, and per camera
    cov = defaultdict(lambda: [0, set()])
    for it in items:
        for key in ((it["event"], it["gravity"]), ("camera", it["camera_id"]), ("gravity", it["gravity"])):
            cov[key][0] += 1
            cov[key][1].add(it["physics_group"])
    lines = ["| group | items | distinct physics |", "|---|---:|---:|"]
    lines += [f"| {' '.join(map(str, k))} | {n} | {len(g)} |" for k, (n, g) in sorted(cov.items(), key=lambda kv: str(kv[0]))]
    (args.out / "coverage.md").write_text("\n".join(lines) + "\n")

    # data quality
    index = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.root / "metadata/trajectories.jsonl")}
    quality = {"windows_excluded": dict(reasons), "trajectories": len(headers),
               "distinct_physics": len(set(groups.values())), "issues": defaultdict(list)}
    by_group = defaultdict(list)
    for sid, g in groups.items():
        by_group[g].append(sid)
    quality["duplicate_groups"] = Counter(len(v) for v in by_group.values() if len(v) > 1)
    for sid, h in headers.items():
        if sid in index and [index[sid]["render"]["width"], index[sid]["render"]["height"]] != list(h["resolution"]):
            quality["issues"]["index resolution differs from metadata and video"].append(sid)
        missing = [c for c in ("cam01", "cam02", "cam03") if c in h["videos"] and not h["camera"].get(c)]
        if missing:
            quality["issues"]["camera parameters not recorded: " + ",".join(missing)].append(sid)
        if len(h["videos"]) < 3:
            quality["issues"]["fewer than three videos"].append(sid)
        for issue in h.get("camera_issues", []):
            quality["issues"][issue.split(" (")[0] if "image offset" in issue else issue].append(sid)
    failed = args.compact.parent / f"{args.compact.name}_failed.json"
    if failed.exists():
        for sid, why in json.loads(failed.read_text()).items():
            quality["issues"]["metadata unreadable: " + why.split(":")[0]].append(sid)
    quality["issues"] = {k: {"count": len(v), "examples": v[:5]} for k, v in quality["issues"].items()}
    quality["duplicate_groups"] = {str(k): v for k, v in sorted(quality["duplicate_groups"].items())}
    (args.out / "quality.json").write_text(json.dumps(quality, indent=1))
    print(f"{len(items)} items from {len(headers)} trajectories; excluded windows {dict(reasons)}")


if __name__ == "__main__":
    main()
