"""Gravity-editing generation inputs for PhyEditing, and scoring of the generated videos.

`build`: for benchmark items, the condition frame (the last frame of the static prelude, identical across the five
gravities of a scene), a prompt (event description + target gravity) and a generation template row in the format
the physedit runners read.

`score`: generated videos tracked from their first frame (= the condition frame) are fitted in the item's declared
window. Frame times are the condition time plus frame index / model fps. The declared initial state is the reference
trajectory's at the target gravity (as for the ground-truth videos); `--v0 free` leaves the launch velocity open, so a
flight that starts earlier or later than in the reference is fitted by its curvature alone.

    python scripts/phyediting_gen.py build ITEMS.jsonl OUT_DIR --host-root DIR [--per-gravity N] [--seed S]
    python scripts/phyediting_gen.py rows TEMPLATE.jsonl ROWS.jsonl --model ltx_i2v --frames 81 --fps 16 --out-root DIR
    python scripts/phyediting_gen.py jobs ITEMS.jsonl ROWS.jsonl JOBS.jsonl      (SAM2 jobs for the generated videos)
    python scripts/phyediting_gen.py score ITEMS.jsonl ROWS.jsonl TRACKS.jsonl OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gravtrace.fit import safe_fit  # noqa: E402

CONDITION_FRAME = 23  # the prelude is static for 24 frames, so this frame is the same at every gravity
EVENTS = {
    "T03": "On a table, a thin book slides into a soda can standing at the table edge; the can falls off onto a lower shelf and a phone drops to the floor.",
    "T05": "A soda can slides across a table toward two objects with a gap between them.",
    "T08": "On a small table, a moving book hits another book that carries a mug; the mug and the books slide off the table edge and fall to the floor.",
    "T16": "A folder leaning on a book tips over when its support is removed and falls onto a small box.",
    "T17": "A red soda can rolls down a tilted book used as a ramp on a table, rolls past a tissue box and falls off the table edge.",
    "T18": "A book is pulled out of a stack of books on a table; the books above lose their support, tilt and tumble down.",
    "T19": "A soda can slides across a table and hits another can, which is pushed to the table edge and falls.",
    "T20": "A sliding book pushes a soda can, which pushes a small box, which pushes a cup, in a chain reaction on a table.",
}
GRAVITY = {
    1.62: "The scene takes place under the Moon's low gravity (1.62 m/s^2), so falling objects drop slowly.",
    3.71: "The scene takes place under Mars gravity (3.71 m/s^2), weaker than on Earth, so objects fall more slowly than usual.",
    9.81: "The scene takes place under normal Earth gravity (9.81 m/s^2).",
    15.0: "The scene takes place under strong gravity (15 m/s^2), about 1.5 times Earth's, so objects fall faster than usual.",
    24.0: "The scene takes place under very strong gravity (24 m/s^2), about 2.5 times Earth's, so objects fall very fast.",
}
NEGATIVE = "camera motion, cuts, blurry, distorted objects, objects appearing or disappearing, text, watermark"


def build(args) -> None:
    items = [json.loads(line) for line in open(args.items)]
    # scenes (background, camera and physics setup without gravity) seen at several gravities; one item per gravity
    by_scene = defaultdict(dict)
    for it in items:
        if it["event"] not in EVENTS:
            continue
        scene = (it["trajectory"].replace(f"g0{[1.62, 3.71, 9.81, 15.0, 24.0].index(it['gravity']) + 1}_", "g?_"), it["camera_id"])
        by_scene[scene].setdefault(it["gravity"], it)
    rng = random.Random(args.seed)
    scenes = sorted(by_scene, key=lambda s: (-len(by_scene[s]), s))
    rng.shuffle(scenes)
    scenes.sort(key=lambda s: -len(by_scene[s]))
    chosen, per_g = [], defaultdict(int)
    for s in scenes:
        if all(per_g[g] >= args.per_gravity for g in GRAVITY):
            break
        for g, it in by_scene[s].items():
            if per_g[g] < args.per_gravity:
                chosen.append(it)
                per_g[g] += 1
    out = Path(args.out)
    (out / "conditioning").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    rows = []
    for k, it in enumerate(chosen):
        sid = it["id"].replace("__", "-")[:120] + f"-{k:03d}"
        cap = cv2.VideoCapture(str(Path(args.data_root) / it["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, CONDITION_FRAME)
        ok, img = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(out / "conditioning" / f"{sid}.png"), img)
        pdir = out / "prompts" / sid
        pdir.mkdir(exist_ok=True)
        (pdir / "prompt.txt").write_text(f"{EVENTS[it['event']]} The camera is fixed. {GRAVITY[it['gravity']]}")
        (pdir / "negative_prompt.txt").write_text(NEGATIVE)
        h, w = img.shape[:2]
        host = args.host_root.rstrip("/")
        rows.append({
            "index": len(rows), "mode": "i2v", "sample_id": sid, "source_sample_id": sid, "item_id": it["id"],
            "benchmark": f"phyediting_{it['event']}", "scenario": "projectile",
            "source_video": f"{args.source_root.rstrip('/')}/{it['video']}", "source_width": w, "source_height": h,
            "source_sample_fps": float(it["fps"]), "source_native_fps": float(it["fps"]),
            "source_orientation": "landscape" if w > h else "portrait", "aspect_ratio": "16:9",
            "condition_image": f"{host}/conditioning/{sid}.png", "condition_frames": 1,
            "condition_frame_index": CONDITION_FRAME, "condition_time_ms": 1000.0 * CONDITION_FRAME / it["fps"],
            "prompt_file": f"{host}/prompts/{sid}/prompt.txt", "negative_prompt_file": f"{host}/prompts/{sid}/negative_prompt.txt",
            "prompt_granularity": "fine", "seed": args.seed, "gravity_truth": it["gravity"], "protocol": "phyediting_gravity_v1",
            "gen_height": 480, "gen_width": 704 if w > h else 480,
        })
    with open(out / "template.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(out / "videos.txt", "w") as f:
        f.writelines(it["video"] + "\n" for it in chosen)
    print(len(rows), "rows;", dict(per_g))


def model_rows(args) -> None:
    """A model's manifest from the template: frame count, frame rate and output paths under OUT_ROOT/<model>."""
    with open(args.out, "w") as f:
        for line in open(args.template):
            row = json.loads(line)
            out_dir = f"{args.out_root.rstrip('/')}/{args.model}/{row['sample_id']}"
            row.update(model_name=args.model, pipeline=args.model, num_frames=args.frames, output_fps=args.fps, out_dir=out_dir,
                       generated_video=f"{out_dir}/raw.mp4", generated_video_resized_to_source=f"{out_dir}/resized.mp4")
            f.write(json.dumps(row) + "\n")


def track_jobs(args) -> None:
    """SAM2 jobs for generated videos: every movable object prompted with its declared box in the condition frame."""
    from phyediting_samples import prompt_boxes
    items = {json.loads(l)["id"]: json.loads(l) for l in open(args.items)}
    with open(args.out, "w") as f:
        for line in open(args.rows):
            row = json.loads(line)
            it = items[row["item_id"]]
            header = json.loads((Path(args.compact) / f"{it['trajectory']}.json").read_text())
            with np.load(Path(args.compact) / f"{it['trajectory']}.npz") as z:
                boxes = prompt_boxes(header, z, it["camera_id"], frame=row["condition_frame_index"])
            video = row["generated_video_resized_to_source"].replace(args.host_prefix, args.local_prefix) if args.local_prefix else row["generated_video_resized_to_source"]
            f.write(json.dumps({"id": row["sample_id"], "video": video, "prompt_frame": 0, "objects": boxes,
                                "last_frame": int(row["num_frames"]) - 1}) + "\n")


def score(args) -> None:
    items = {json.loads(l)["id"]: json.loads(l) for l in open(args.items)}
    tracks = {json.loads(l)["id"]: json.loads(l) for l in open(args.tracks)}
    out = open(args.out, "w")
    for line in open(args.rows):
        row = json.loads(line)
        it = items[row["item_id"]]
        tr = tracks.get(row["sample_id"], {}).get("tracks", {}).get(it["object_name"])
        fps_gen = float(row.get("output_fps") or tracks.get(row["sample_id"], {}).get("fps") or 16.0)
        t_cond = row["condition_frame_index"] / float(it["fps"])
        w0, n = it["window"]["start_frame"] / float(it["fps"]), it["window"]["max_frames"]
        w1 = (it["window"]["start_frame"] + n - 1) / float(it["fps"])
        frames, boxes, times = [], [], []
        for f, b in zip(tr["frames"] if tr else [], tr["xyxy"] if tr else []):
            t = t_cond + f / fps_gen
            if w0 - 1e-6 <= t <= w1 + 1e-6:
                frames.append(int(round(t * it["fps"])))
                boxes.append(b)
                times.append(t - w0)
        motion = dict(it["motion"])
        if times:  # GravTrace times the first fitted frame from the declared state by t0
            motion["t0"] = [times[0], times[0]]
        if args.v0 == "free":  # robustness variant: launch speed within +-50 %, direction free
            speed = float(np.linalg.norm(motion.pop("v0")))
            motion.update(speed=[0.5 * speed, 1.5 * speed + 0.05], angle_deg=[-89.0, 89.0],
                          directions=[(np.asarray(it["motion"]["v0"]) / max(speed, 1e-9)).tolist()])
        # the declared visibility of each edge, at the reference frame nearest in time
        seen = it["window"].get("edges_visible", {})
        window = {"max_frames": len(frames), "start_frame": 0,
                  "edges_visible": {str(i): seen.get(str(f), [True] * 4) for i, f in enumerate(frames)}}
        sample = {"id": row["sample_id"], "scenario": "projectile", "fps": it["fps"], "image_size": it["image_size"],
                  "camera": it["camera"], "object": it["object"], "motion": motion, "window": window,
                  "boxes": {"frames": list(range(len(frames))), "xyxy": boxes, "times": times}}
        pred = safe_fit(sample) if len(frames) >= 4 else {"id": row["sample_id"], "status": "no_track", "frames": len(frames)}
        out.write(json.dumps({**pred, "item_id": row["item_id"], "gravity_target": row["gravity_truth"]}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("items")
    b.add_argument("out")
    b.add_argument("--host-root", required=True)
    b.add_argument("--data-root", default="data/phyediting")
    b.add_argument("--source-root", required=True, help="where the reference videos are on the generation host")
    b.add_argument("--per-gravity", type=int, default=8)
    b.add_argument("--seed", type=int, default=20261008)
    s = sub.add_parser("score")
    s.add_argument("items")
    s.add_argument("rows")
    s.add_argument("tracks")
    s.add_argument("out")
    s.add_argument("--v0", default="declared", choices=["declared", "free"])
    m = sub.add_parser("rows")
    m.add_argument("template")
    m.add_argument("out")
    m.add_argument("--model", required=True)
    m.add_argument("--frames", type=int, default=81)
    m.add_argument("--fps", type=float, default=16.0)
    m.add_argument("--out-root", required=True)
    t = sub.add_parser("jobs")
    t.add_argument("items")
    t.add_argument("rows")
    t.add_argument("out")
    t.add_argument("--compact", default="data/compact")
    t.add_argument("--host-prefix", default="")
    t.add_argument("--local-prefix", default="")
    args = ap.parse_args()
    {"build": build, "score": score, "rows": model_rows, "jobs": track_jobs}[args.cmd](args)


if __name__ == "__main__":
    main()
