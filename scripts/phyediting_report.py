"""Error tables for gravity estimates on PhyEditing windows.

A window is evaluated when the object is in view in >= 4 frames that each show >= 2 box edges not hidden behind
another declared object, gravity moves it visibly (its sag, the largest shift gravity causes on a visible box edge
relative to the same declared flight without gravity, is >= --sag px) and the tracker
follows it (SAM2 boxes cover >= 80 % of the visible frames and no edge the declared scene leaves visible differs
from the projected true box by more than --rms px once a constant offset is removed). Per video, the window with the largest sag is reported.

    python scripts/phyediting_report.py ORACLE.jsonl OBSERVED.jsonl METHOD=PRED.jsonl ... [--sag 40] [--rms 4]
        [--by event gravity camera] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gravtrace.camera import Camera  # noqa: E402


def visible_sag(s: dict) -> tuple[float, int]:
    """(largest displacement gravity causes on an edge the declared scene leaves visible, px; frames with >= 2 visible
    edges). The image box is that of the declared corners on the declared flight with and without gravity."""
    from gravtrace.camera import quat_to_matrix
    cam = Camera.from_dict(s["camera"], s["image_size"])
    n = s["window"]["max_frames"]
    frames = np.arange(s["window"]["start_frame"], s["window"]["start_frame"] + n)
    t = (frames - frames[0]) / s["fps"]
    corners = np.asarray(s["object"]["corners"]) @ quat_to_matrix(s["object"].get("quaternion")).T
    x0, v0 = np.asarray(s["object"]["position"]), np.asarray(s["motion"].get("v0", [0, 0, 0]))
    gd = np.asarray(s["motion"]["gravity_dir"]) * s["truth"]["gravity"]

    def box(path):
        uv = cam.project(path[:, None, :] + corners[None])
        return np.hstack([uv.min(axis=1), uv.max(axis=1)])
    delta = np.abs(box(x0 + np.outer(t, v0) + 0.5 * np.outer(t * t, gd)) - box(x0 + np.outer(t, v0)))
    seen = np.array([s["window"].get("edges_visible", {}).get(str(f), [True] * 4) for f in frames], dtype=bool)
    in_view = np.isin(frames, s["boxes"]["frames"])[:, None]
    usable = seen & in_view
    return (float(np.nanmax(np.where(usable, delta, 0.0))) if usable.any() else 0.0), int((usable.sum(axis=1) >= 2).sum())


def sag_px(s: dict) -> float:
    cam = Camera.from_dict(s["camera"], s["image_size"])
    frames = np.asarray(s["boxes"]["frames"], dtype=int)
    if len(frames) < 2:
        return 0.0
    t = (frames - s["window"]["start_frame"]) / s["fps"]
    x0, v0 = np.asarray(s["object"]["position"]), np.asarray(s["motion"].get("v0", [0, 0, 0]))
    gd = np.asarray(s["motion"]["gravity_dir"]) * s["truth"]["gravity"]
    with_g = cam.project(x0 + np.outer(t, v0) + 0.5 * np.outer(t * t, gd))
    without = cam.project(x0 + np.outer(t, v0))
    return float(np.nanmax(np.linalg.norm(with_g - without, axis=1)))


def tracking(oracle: dict, observed: dict) -> tuple[float, float]:
    """(coverage of visible frames, largest |observed - true| edge deviation in px over the edges the declared scene
    leaves visible, after removing each edge's median offset)."""
    true = {f: np.asarray(b) for f, b in zip(oracle["boxes"]["frames"], oracle["boxes"]["xyxy"])}
    seen = oracle["window"].get("edges_visible", {})
    pairs = [(np.asarray(b) - true[f], np.asarray(seen.get(str(f), [True] * 4), dtype=bool))
             for f, b in zip(observed["boxes"]["frames"], observed["boxes"]["xyxy"]) if f in true]
    if not pairs:
        return 0.0, float("inf")
    diff = np.array([d for d, _ in pairs])
    mask = np.array([m for _, m in pairs])
    diff = np.where(mask, diff, np.nan)
    dev = np.abs(diff - np.nanmedian(diff, axis=0))
    return len(pairs) / len(true), float(np.nanmax(dev)) if np.isfinite(dev).any() else float("inf")


def stats(errors: list[float], failed: int) -> dict:
    e = np.asarray(errors)
    out = {"n": len(errors) + failed, "failed": failed}
    if e.size:
        out.update(mean=float(e.mean()), median=float(np.median(e)), p95=float(np.percentile(e, 95)), max=float(e.max()),
                   within_10pct=float((e <= 0.10).mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle")
    ap.add_argument("observed")
    ap.add_argument("preds", nargs="+", help="METHOD=PRED.jsonl")
    ap.add_argument("--sag", type=float, default=40.0)
    ap.add_argument("--rms", type=float, default=4.0)
    ap.add_argument("--cover", type=float, default=0.8)
    ap.add_argument("--by", nargs="*", default=["event", "gravity", "camera"])
    ap.add_argument("--sens", type=float, default=0.0, help="also require this GravTrace sensitivity (px) on the true boxes")
    ap.add_argument("--oracle-pred", help="GravTrace predictions on the true boxes (for --sens)")
    ap.add_argument("--json")
    args = ap.parse_args()
    oracle = {}
    for line in open(args.oracle):
        s = json.loads(line)
        oracle[s["id"]] = s
    observed = {}
    for line in open(args.observed):
        s = json.loads(line)
        observed[s["id"]] = s
    sensitivity = {}
    if args.oracle_pred:
        for line in open(args.oracle_pred):
            p = json.loads(line)
            sensitivity[p["id"]] = p.get("sensitivity_px", 0.0) if p.get("status") == "ok" else 0.0
    windows, excluded = {}, defaultdict(int)
    for wid, o in oracle.items():
        if len(o["boxes"]["frames"]) < 4:
            excluded["not visible"] += 1
            continue
        sag, usable_frames = visible_sag(o)
        if usable_frames < 4:
            excluded["occluded"] += 1
            continue
        if sag < args.sag:
            excluded["small sag"] += 1
            continue
        if args.sens and sensitivity.get(wid, 0.0) < args.sens:
            excluded["low sensitivity"] += 1
            continue
        cover, rms = tracking(o, observed[wid]) if wid in observed else (0.0, float("inf"))
        if cover < args.cover or rms > args.rms:
            excluded["not trackable"] += 1
            continue
        windows[wid] = sensitivity.get(wid, sag) if args.sens else sag  # per video, the best-constrained window
    best = {}  # per video: the window with the largest sag
    for wid, sag in windows.items():
        key = (oracle[wid]["gt_id"], oracle[wid]["meta"]["camera"])
        if key not in best or sag > windows[best[key]]:
            best[key] = wid
    chosen = set(best.values())
    report = {"windows_evaluated": len(windows), "videos": len(chosen), "excluded_windows": dict(excluded), "methods": {}}
    for spec in args.preds:
        method, path = spec.split("=", 1)
        preds = {}
        for line in open(path):
            p = json.loads(line)
            preds[p["id"]] = p
        groups: dict = defaultdict(lambda: ([], [0]))
        for wid in chosen:
            o, p = oracle[wid], preds.get(wid, {})
            g = o["truth"]["gravity"]
            keys = [("all",)] + [(k, o["meta"][k]) for k in args.by]
            for key in keys:
                if p.get("status") == "ok":
                    groups[key][0].append(abs(p["gravity"] - g) / g)
                else:
                    groups[key][1][0] += 1
        report["methods"][method] = {" ".join(map(str, k)): stats(e, f[0]) for k, (e, f) in sorted(groups.items(), key=lambda kv: str(kv[0]))}
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    print(f"videos {report['videos']} (windows {report['windows_evaluated']}), excluded windows {report['excluded_windows']}")
    for method, groups in report["methods"].items():
        a = groups["all"]
        print(f"{method:28s} n={a['n']:4d} fail={a['failed']:3d} mean {a.get('mean', 0) * 100:6.2f}% median {a.get('median', 0) * 100:6.2f}% "
              f"p95 {a.get('p95', 0) * 100:6.1f}% max {a.get('max', 0) * 100:6.1f}%")


if __name__ == "__main__":
    main()
