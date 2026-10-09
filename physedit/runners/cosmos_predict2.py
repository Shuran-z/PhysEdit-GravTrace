#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--cosmos-tools", default="/data1/zhangshuran/cosmos_predict2_tools")
    parser.add_argument("--python", default="/data1/zhangshuran/miniconda3/envs/cosmos-predict2/bin/python")
    parser.add_argument("--model-dir", default="/data1/zhangshuran/models/Cosmos-Predict2-2B-Video2World")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.generation_manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = rows[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    status_path = manifest.parent / "cosmos_generation_status.jsonl"
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    completed = skipped = failed = 0
    with status_path.open("a", encoding="utf-8") as status_handle:
        for row in selected:
            start = time.time()
            out_dir = Path(row["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "cosmos_generate.log"
            resized_path = Path(row["generated_video_resized_to_source"])
            if args.skip_existing and resized_path.exists() and resized_path.stat().st_size > 0:
                skipped += 1
                _write_status(status_handle, row, "skipped_existing", start, log_path, resized_path)
                continue

            command = [
                args.python,
                "run_cosmos_v2w_from_video.py",
                "--video_path",
                row["source_video"],
                "--out_dir",
                str(out_dir),
                "--model_dir",
                args.model_dir,
                "--model_size",
                "2B",
                "--resolution",
                "480",
                "--fps",
                str(int(row.get("cosmos_output_fps", 16))),
                "--aspect_ratio",
                str(row.get("aspect_ratio") or "9:16"),
                "--num_frames",
                "81",
                "--num_conditional_frames",
                str(int(row.get("cosmos_predict2_condition_frames", 5))),
                "--sample_fps",
                str(float(row.get("source_sample_fps", 240.0))),
                "--seed",
                str(42 + int(row.get("index", 0))),
                "--guidance",
                "7.0",
                "--prompt_file",
                row["prompt_file"],
                "--negative_prompt_file",
                row["negative_prompt_file"],
                "--tokenizer_path",
                str(Path(args.model_dir) / "tokenizer" / "tokenizer.pth"),
                "--text_encoder_path",
                str(Path(args.model_dir) / "text_encoder"),
            ]
            if args.dry_run:
                print(" ".join(command))
                _write_status(status_handle, row, "dry_run", start, log_path, resized_path)
                continue

            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(command, cwd=args.cosmos_tools, env=env, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                failed += 1
                _write_status(status_handle, row, "failed_generation", start, log_path, resized_path, returncode=proc.returncode)
                continue

            raw_generated = Path(row["generated_video"])
            try:
                _resize_to_source(raw_generated, Path(row["source_video"]), resized_path, fps=float(row.get("cosmos_output_fps", 16.0)))
            except Exception as exc:
                failed += 1
                _write_status(status_handle, row, "failed_resize", start, log_path, resized_path, error=repr(exc))
                continue
            completed += 1
            _write_status(status_handle, row, "completed", start, log_path, resized_path)

    summary = {
        "manifest": str(manifest),
        "selected": len(selected),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "status_path": str(status_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


def _write_status(handle, row, status, start_time, log_path, resized_path, **extra) -> None:
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - start_time, 3),
        "sample_id": row.get("sample_id"),
        "source_video": row.get("source_video"),
        "out_dir": row.get("out_dir"),
        "log_path": str(log_path),
        "generated_video_resized_to_source": str(resized_path),
        "status": status,
        **extra,
    }
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()
    print(json.dumps(payload, sort_keys=True))


def _resize_to_source(generated: Path, source: Path, output: Path, fps: float) -> None:
    width, height = _video_size(source)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"could not read source dimensions: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(generated),
        "-vf",
        f"scale={width}:{height}",
        "-r",
        str(fps),
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def _video_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return width, height


if __name__ == "__main__":
    raise SystemExit(main())
