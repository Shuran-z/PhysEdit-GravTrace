#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--startup-reserve-mib", type=int, default=1536)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.generation_manifest).resolve()
    repo = Path(args.repo_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = rows[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    pending = []
    skipped = 0
    for row in selected:
        output = Path(row["generated_video_resized_to_source"])
        if args.skip_existing and output.exists() and output.stat().st_size > 0:
            skipped += 1
        else:
            pending.append(row)

    settings = {
        "resolution": [576, 1024],
        "video_length": 16,
        "output_fps": 8,
        "frame_stride": 10,
        "ddim_steps": args.ddim_steps,
        "guidance_scale": 7.5,
        "timestep_spacing": "uniform_trailing",
        "guidance_rescale": 0.7,
        "perframe_ae": True,
    }
    if args.dry_run:
        print(json.dumps({"selected": len(selected), "pending": len(pending), "skipped": skipped, "settings": settings}, indent=2))
        return 0

    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    inference = repo / "scripts/evaluation/inference.py"
    config = repo / "configs/inference_1024_v1.0.yaml"
    if not inference.is_file() or not config.is_file():
        parser.error(f"invalid DynamiCrafter repository: {repo}")

    reservation = None
    if args.startup_reserve_mib > 0:
        import torch

        reservation = torch.empty(
            args.startup_reserve_mib * 1024 * 1024,
            dtype=torch.uint8,
            device="cuda:0",
        )

    run_root = manifest.parent / "dynamicrafter_i2v_stage"
    run_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in pending:
        grouped[int(row.get("seed") or 42)].append(row)

    status_path = manifest.parent / "dynamicrafter_i2v_generation_status.jsonl"
    completed = failed = 0
    with status_path.open("a", encoding="utf-8") as status_handle:
        for seed, group in sorted(grouped.items()):
            group_id = f"seed_{seed}_{int(time.time())}"
            input_dir = run_root / group_id / "inputs"
            result_dir = run_root / group_id / "results"
            input_dir.mkdir(parents=True, exist_ok=True)
            prompts = []
            staged = []
            for index, row in enumerate(group):
                source = Path(row["condition_image"])
                suffix = source.suffix.lower() if source.suffix.lower() in {".png", ".jpg", ".jpeg"} else ".png"
                stem = f"{index:05d}_{_safe_id(str(row.get('sample_id') or index))}"
                image_path = input_dir / f"{stem}{suffix}"
                shutil.copy2(source, image_path)
                prompts.append(Path(row["prompt_file"]).read_text(encoding="utf-8").strip())
                staged.append((row, stem))
            (input_dir / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")

            command = [
                sys.executable,
                str(inference),
                "--seed", str(seed),
                "--ckpt_path", str(checkpoint),
                "--config", str(config),
                "--savedir", str(result_dir),
                "--n_samples", "1",
                "--bs", "1",
                "--height", "576",
                "--width", "1024",
                "--unconditional_guidance_scale", "7.5",
                "--ddim_steps", str(args.ddim_steps),
                "--ddim_eta", "1.0",
                "--prompt_dir", str(input_dir),
                "--text_input",
                "--video_length", "16",
                "--frame_stride", "10",
                "--timestep_spacing", "uniform_trailing",
                "--guidance_rescale", "0.7",
                "--perframe_ae",
            ]
            log_path = run_root / group_id / "official_inference.log"
            env = os.environ.copy()
            # The scheduler already narrows the physical GPU namespace. Keep
            # that mapping so logical cuda:0 does not escape back to GPU 0.
            env.setdefault("CUDA_VISIBLE_DEVICES", str(args.cuda_visible_devices))
            with log_path.open("w", encoding="utf-8") as log_handle:
                result = subprocess.run(command, cwd=repo, env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=False)

            for row, stem in staged:
                started = time.time()
                official_output = result_dir / "samples_separate" / f"{stem}_sample0.mp4"
                raw_output = Path(row["generated_video"])
                resized_output = Path(row["generated_video_resized_to_source"])
                try:
                    if result.returncode != 0:
                        raise RuntimeError(f"official inference exited {result.returncode}; see {log_path}")
                    if not official_output.is_file() or official_output.stat().st_size == 0:
                        raise RuntimeError(f"official output missing: {official_output}")
                    raw_output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(official_output, raw_output)
                    _resize_to_source(raw_output, Path(row["source_video"]), resized_output, float(row.get("output_fps") or 8.0), dims=(int(row["source_width"]), int(row["source_height"])))
                    status = "completed"
                    completed += 1
                    error = None
                except Exception as exc:
                    status = "failed"
                    failed += 1
                    error = repr(exc)
                payload = {
                    "sample_id": row.get("sample_id"),
                    "status": status,
                    "seed": seed,
                    "method": "dynamicrafter_official_1024_i2v",
                    "elapsed_seconds": time.time() - started,
                    "output": str(resized_output),
                    "official_log": str(log_path),
                }
                if error:
                    payload["error"] = error
                status_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                status_handle.flush()

    print(json.dumps({"selected": len(selected), "completed": completed, "skipped": skipped, "failed": failed, "settings": settings}, indent=2))
    return 1 if failed else 0


def _resize_to_source(raw_path: Path, source_video: Path, output_path: Path, fps: float, dims=None) -> None:
    probe = None if dims else subprocess.run(  # public211: dims from the manifest
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(source_video)],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = dims if dims else [int(value) for value in probe.stdout.strip().split("x")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw_path), "-vf", f"scale={width}:{height}:flags=lanczos", "-r", f"{fps:g}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)[:80]


if __name__ == "__main__":
    raise SystemExit(main())
