"""Compact per-trajectory records from every PhyEditing release.

The releases store the same kind of record in different layouts:
  canonical subset   metadata/trajectories.jsonl -> data/<group>/<event>/metadata/<id>__trajectory.json.gz (cam01 only)
  formal archives    .../<id>__cam01.json.gz, with each camera's view in <id>__camNN.json.gz or configs/<id>/camNN.json
Each trajectory becomes `<id>.npz` (dynamic objects' states per frame) and `<id>.json` (cameras, objects,
gravity, timing, videos, source).

    python scripts/phyediting_extract.py DATA_ROOT OUT_DIR [--workers N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STATE_KEYS = ("position", "quaternion_xyzw", "velocity", "angular_velocity")
CAMERAS = ("cam01", "cam02", "cam03")
OBJECT_KEYS = ("asset_id", "category", "shape", "size", "pose", "fixed", "motion", "visual_mesh_path", "collision_mesh_path",
               "mass", "density", "friction", "restitution", "scale", "linear_damping", "angular_damping", "collision_proxy",
               "collision_proxy_size")


def load_json(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def discover(root: Path) -> list[dict]:
    """One job per trajectory: id, physics metadata, videos and where each camera's view is recorded.

    Physics records live in `metadata/` directories (`<id>__cam01.json[.gz]`); a camera's view comes from that
    camera's own metadata record, else from `configs/<id>__camNN.json` or `configs/<id>/camNN.json`."""
    videos, metas, views, configs = {}, {}, {}, {}
    pattern = re.compile(r"(.+)__(cam0[123])\.json(\.gz)?$")
    for dirpath, _, names in os.walk(root):
        parent = os.path.basename(dirpath)
        if os.path.basename(os.path.dirname(dirpath)) == "metadata":  # metadata/<id>/<id>__cam01.json.gz
            parent = "metadata"
        for name in names:
            path = os.path.join(dirpath, name)
            if name.endswith(".mp4"):
                videos[name[:-4]] = path
            elif (m := pattern.match(name)) and parent in ("metadata", "configs"):
                sid, cam = m.group(1), m.group(2)
                if parent == "metadata" and cam == "cam01":
                    metas[sid] = path
                # the camera's own render record is authoritative; configs can be stale (T08's differ from its videos)
                if parent == "configs":
                    configs[(sid, cam)] = path
                if parent == "metadata" or (sid, cam) not in views:
                    views[(sid, cam)] = path
            elif parent != "configs" and re.fullmatch(r"cam0[123]\.json", name) and os.path.basename(os.path.dirname(dirpath)) == "configs":
                views.setdefault((parent, name[:5]), path)
                configs[(parent, name[:5])] = path
    jobs = []
    for line in open(root / "metadata/trajectories.jsonl"):
        row = json.loads(line)
        jobs.append({"id": row["sample_id"], "source": "canonical", "meta": str(root / row["trajectory_metadata"]),
                     "videos": dict(row["videos"]), "views": {"cam01": None}})
    for sid, path in metas.items():
        rel = Path(path).relative_to(root)
        jobs.append({"id": sid, "source": "/".join(rel.parts[:2]), "meta": path,
                     "videos": {c: os.path.relpath(videos[f"{sid}__{c}"], root) for c in CAMERAS if f"{sid}__{c}" in videos},
                     "views": {c: views.get((sid, c)) for c in CAMERAS},
                     "configs": {c: configs.get((sid, c)) for c in CAMERAS}})
    return jobs


def extract(job: dict, out_dir: Path) -> str:
    meta = load_json(job["meta"])
    names = [o["name"] for o in meta["dynamic_objects"]]
    frames = meta["frames"]
    arrays = {k: np.full((len(frames), len(names), 4 if k == "quaternion_xyzw" else 3), np.nan) for k in STATE_KEYS}
    for i, frame in enumerate(frames):
        for obj in frame["objects"]:
            if obj["name"] in names:
                j = names.index(obj["name"])
                for k in STATE_KEYS:
                    if k in obj:
                        arrays[k][i, j] = obj[k]
    exp, render = meta["experiment"], meta["render"]
    cameras, issues = {}, []
    for cam, view_path in job["views"].items():
        if view_path is None and cam == "cam01":
            cameras[cam] = render.get("fixed_view")
        elif view_path:
            cameras[cam] = load_json(view_path)["render"].get("fixed_view")
    # a camera known only from configs is trusted when this sample's cam01 config matches what the renderer recorded
    configs = job.get("configs", {})
    if configs.get("cam01") and cameras.get("cam01"):
        cfg01 = load_json(configs["cam01"])["render"].get("fixed_view") or {}
        rec01 = cameras["cam01"]
        same = all(abs(a - b) < 1e-6 for a, b in zip(cfg01.get("position", []) + cfg01.get("look_at", []) + [cfg01.get("fov_deg", 0)],
                                                        rec01["position"] + rec01["look_at"] + [rec01["fov_deg"]]))
        if not same:
            issues.append("configs differ from the rendered cam01")
            for cam in ("cam02", "cam03"):
                if job["views"].get(cam) and job["views"][cam] == configs.get(cam):  # only a config: unverified
                    cameras[cam] = None
                    issues.append(f"{cam} view known only from stale configs")
    gvec = exp.get("gravity_vector") or meta.get("scene_overrides", {}).get("gravity")
    header = {
        "sample_id": job["id"], "source": job["source"], "event": exp.get("template_id"), "background": exp.get("background_id"),
        "gravity": -float(gvec[2]), "gravity_vector": gvec, "parameter": exp.get("parameter"), "outcome": exp.get("outcomes"),
        "fps": render["fps"], "resolution": render["resolution"], "pre_static_frames": render.get("pre_static_frames"),
        "camera": cameras,
        "objects": [{"name": o["name"], **{k: o["spec"].get(k) for k in OBJECT_KEYS}} for o in meta["dynamic_objects"]],
        "surface_z": exp.get("surface_z"), "event_sequence": exp.get("event_sequence"),
        "physics": {k: meta.get("scene_overrides", {}).get(k) for k in ("gravity", "dt", "substeps")},
        "time": meta.get("time_diagnostic"), "videos": job["videos"],  # relative to the data root
        "camera_issues": issues,
    }
    out = out_dir / f"{job['id']}.npz"
    np.savez_compressed(out, names=np.array(names), **arrays)
    out.with_suffix(".json").write_text(json.dumps(header))
    return job["id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = discover(args.root)
    todo = [j for j in jobs if os.path.exists(j["meta"]) and not (args.out / f"{j['id']}.json").exists()]
    print(f"{len(jobs)} trajectories found, {len(todo)} to extract", file=sys.stderr)
    failed, failures = 0, {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(extract, j, args.out): j["id"] for j in todo}
        for n, (fut, sid) in enumerate(futures.items(), 1):
            try:
                fut.result()
            except Exception as exc:  # report and continue: one malformed record must not stop the rest
                failed += 1
                failures[sid] = f"{type(exc).__name__}: {exc}"
                print(f"FAILED {sid}: {failures[sid]}", file=sys.stderr)
            if n % 500 == 0:
                print(f"{n}/{len(todo)}", file=sys.stderr)
    if failures:
        (args.out.parent / f"{args.out.name}_failed.json").write_text(json.dumps(failures, indent=1))
    print(f"done, {failed} failed", file=sys.stderr)


if __name__ == "__main__":
    main()
