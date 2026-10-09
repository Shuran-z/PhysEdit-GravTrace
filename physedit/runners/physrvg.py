#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import torch
from accelerate.utils import set_seed
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from peft import PeftModel
from PIL import Image

from fastvideo.models.wan_v2v.model_wan_v2v import WanTransformer3DModel
from fastvideo.models.wan_v2v.pipeline_wan_v2v import WanImageToVideoPipeline


NEGATIVE_PROMPT = (
    "oversaturated, overexposed, static, blurry details, subtitles, text, "
    "painting, gray cast, worst quality, low quality, JPEG artifacts, "
    "deformed objects, duplicated objects, broken geometry, camera motion"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--model-label", default="PhysRVG")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-jsonl", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--startup-reserve-mib", type=int, default=1536)
    parser.add_argument("--offload-mode", choices=("none", "model", "sequential"), default="model")
    parser.add_argument(
        "--condition-policy",
        choices=("source_history5", "static_frame"),
        default="source_history5",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path, num_shards: int, shard_index: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def append_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def resize_to_source(source: Path, destination: Path, width: int, height: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-preset",
            "fast",
            str(destination),
        ],
        check=True,
    )


def load_source_history(source: Path, count: int = 5) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source video: {source}")
    frames: list[Image.Image] = []
    try:
        for index in range(count):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"cannot read declared history frame {index}: {source}")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    return frames


def load_pipeline(args: argparse.Namespace) -> WanImageToVideoPipeline:
    dtype = torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float32
    )
    transformer = WanTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=dtype
    )
    if args.lora_checkpoint is not None:
        transformer = PeftModel.from_pretrained(transformer, args.lora_checkpoint)
        transformer.set_adapter("default")
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_id, transformer=transformer, vae=vae, torch_dtype=dtype
    )
    pipe.vae.enable_slicing()
    if args.offload_mode == "model":
        pipe.enable_model_cpu_offload(gpu_id=args.device)
    elif args.offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload(gpu_id=args.device)
    else:
        pipe.to(torch.device(f"cuda:{args.device}"))
    return pipe


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest, args.num_shards, args.shard_index)
    if args.limit is not None:
        rows = rows[: args.limit]

    device = torch.device(f"cuda:{args.device}")
    # Claim the already-approved idle GPU while the large CPU checkpoint loads.
    # Without this, the card appears empty for several minutes and another job
    # can start before model offload hooks allocate their first CUDA tensors.
    reservation = None
    if args.startup_reserve_mib > 0:
        reservation = torch.empty(
            args.startup_reserve_mib * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
    try:
        pipe = load_pipeline(args)
    except Exception:
        del reservation
        torch.cuda.empty_cache()
        raise

    for row in rows:
        source_sample_id = row["source_sample_id"]
        sample_id = row.get("sample_id") or source_sample_id
        granularity = row["prompt_granularity"]
        out_dir = (
            Path(row["out_dir"])
            if row.get("out_dir")
            else args.output_root / "generated" / source_sample_id / granularity
        )
        raw_path = (
            Path(row["generated_video"])
            if row.get("generated_video")
            else out_dir / "generated_video.mp4"
        )
        resized_path = (
            Path(row["generated_video_resized_to_source"])
            if row.get("generated_video_resized_to_source")
            else out_dir / "generated_video_resized_to_source.mp4"
        )
        if resized_path.is_file() and resized_path.stat().st_size > 0 and not args.overwrite:
            append_status(
                args.status_jsonl,
                {
                    "sample_id": sample_id,
                    "source_sample_id": source_sample_id,
                    "prompt_granularity": granularity,
                    "status": "skipped_existing",
                },
            )
            continue

        prompt_path = Path(row["prompt_file"])
        condition_path = Path(row["condition_image"])
        if not prompt_path.is_file() or not condition_path.is_file():
            raise FileNotFoundError(f"missing prompt or condition for {sample_id}/{granularity}")

        prompt = prompt_path.read_text(encoding="utf-8").strip()
        condition = Image.open(condition_path).convert("RGB")
        if args.condition_policy == "source_history5":
            prefix = load_source_history(Path(row["source_video"]), count=5)
            condition_description = "source frames 0-4; frame 4 is the declared condition image"
            uses_source_video = True
        else:
            prefix = [condition.copy() for _ in range(5)]
            condition_description = "declared condition image repeated for five static prefix frames"
            uses_source_video = False
        seed = int(row["seed"])
        set_seed(seed)
        torch.cuda.empty_cache()
        started = time.time()
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            frames = pipe(
                video=prefix,
                device=device,
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=5,
                do_cfg=False,
            )[0][0]
            export_to_video(frames, raw_path, fps=args.fps)
            resize_to_source(
                raw_path,
                resized_path,
                int(row["source_width"]),
                int(row["source_height"]),
            )
            probe = probe_video(resized_path)
            append_status(
                args.status_jsonl,
                {
                    "sample_id": sample_id,
                    "source_sample_id": source_sample_id,
                    "prompt_granularity": granularity,
                    "status": "complete",
                    "seed": seed,
                    "model_label": args.model_label,
                    "lora_checkpoint": str(args.lora_checkpoint) if args.lora_checkpoint else None,
                    "condition_policy": condition_description,
                    "uses_source_video": uses_source_video,
                    "source_frame_range": [0, 4] if uses_source_video else None,
                    "offload_mode": args.offload_mode,
                    "elapsed_sec": round(time.time() - started, 3),
                    "output": str(resized_path),
                    "probe": probe,
                },
            )
        except Exception as exc:
            append_status(
                args.status_jsonl,
                {
                    "sample_id": sample_id,
                    "source_sample_id": source_sample_id,
                    "prompt_granularity": granularity,
                    "status": "failed",
                    "error": repr(exc),
                },
            )
            raise

    del reservation
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
