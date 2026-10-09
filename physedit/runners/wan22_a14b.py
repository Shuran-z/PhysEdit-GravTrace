#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import av
import imageio_ffmpeg
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--status-jsonl", type=Path, required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--physalign-adapter-dir", type=Path)
    parser.add_argument("--lora-merge-chunk-rows", type=int, default=128)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def probe_video(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty output: {path}")
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise RuntimeError(f"expected one video stream: {path}")
        stream = streams[0]
        frames = sum(1 for _ in container.decode(stream))
        result = {
            "codec_name": stream.codec_context.name,
            "width": stream.width,
            "height": stream.height,
            "r_frame_rate": str(stream.average_rate),
            "nb_read_frames": frames,
            "size": path.stat().st_size,
        }
    if result["codec_name"] != "h264" or result["nb_read_frames"] != 81:
        raise RuntimeError(f"invalid video probe: {result}")
    return result


def resize_to_source(source: Path, destination: Path, width: int, height: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.stem + ".partial.mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )
    subprocess.run(
        [
            ffmpeg,
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
            str(tmp),
        ],
        check=True,
    )
    probe_video(tmp)
    os.replace(tmp, destination)


def write_video_av(video: torch.Tensor, path: Path, fps: int) -> None:
    """Fallback for Wan's save_video, which logs its exceptions at INFO level and can leave a file without a video
    stream (seen for one fixed-seed row: the last frame reached ffmpeg truncated). Same pixel conversion as
    save_video (value range -1..1 to 0..255, truncated), encoded with PyAV; errors propagate."""
    frames = ((video.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "10"}
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.code_dir))

    import wan  # noqa: PLC0415
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # noqa: PLC0415
    from wan.utils.utils import save_video  # noqa: PLC0415

    if args.physalign_adapter_dir is not None:
        from wan22_physalign_lora import install_physalign_merge_hook  # noqa: PLC0415

        install_physalign_merge_hook(
            wan.WanI2V,
            args.physalign_adapter_dir,
            chunk_rows=args.lora_merge_chunk_rows,
        )

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cfg = WAN_CONFIGS["i2v-A14B"]
    torch.cuda.set_device(0)
    atomic_json(
        args.state_json,
        {"state": "loading", "completed": 0, "total": len(rows), "failed": 0},
    )
    pipeline = wan.WanI2V(
        config=cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )
    if args.physalign_adapter_dir is not None:
        summaries = getattr(pipeline, "_physalign_merge_summaries", [])
        if len(summaries) != 2:
            raise RuntimeError(f"expected two merged PhysAlign experts, got {len(summaries)}")
        atomic_json(
            args.state_json.parent / "physalign_merge_audit.json",
            {"state": "PASS", "experts": summaries},
        )

    completed = failed = 0
    for row in rows:
        sample_id = row["sample_id"]
        raw_path = Path(row["generated_video"])
        resized_path = Path(row["generated_video_resized_to_source"])
        try:
            probe = probe_video(resized_path)
            completed += 1
            append_jsonl(
                args.status_jsonl,
                {"sample_id": sample_id, "status": "skipped_existing", "probe": probe},
            )
            continue
        except Exception:
            pass

        atomic_json(
            args.state_json,
            {
                "state": "running",
                "completed": completed,
                "total": len(rows),
                "current": sample_id,
                "failed": failed,
            },
        )
        started = time.time()
        try:
            prompt = Path(row["prompt_file"]).read_text(encoding="utf-8").strip()
            image = Image.open(row["condition_image"]).convert("RGB")
            video = pipeline.generate(
                prompt,
                image,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver="unipc",
                sampling_steps=args.sample_steps,
                guide_scale=cfg.sample_guide_scale,
                seed=int(row["seed"]),
                offload_model=True,
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_tmp = raw_path.with_name(raw_path.stem + ".partial.mp4")
            save_video(
                tensor=video[None],
                save_file=str(raw_tmp),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            try:
                probe_video(raw_tmp)
            except RuntimeError as exc:
                print(f"[runner] save_video left an unreadable file ({exc}); writing it with PyAV", flush=True)
                write_video_av(video, raw_tmp, cfg.sample_fps)
                probe_video(raw_tmp)
            del video
            torch.cuda.empty_cache()
            os.replace(raw_tmp, raw_path)
            resize_to_source(
                raw_path,
                resized_path,
                int(row["source_width"]),
                int(row["source_height"]),
            )
            probe = probe_video(resized_path)
        except Exception as exc:
            # One bad row (e.g. a truncated write) must not stop the run: record it, go on, and exit
            # non-zero at the end so run_node.sh retries just the failed rows.
            failed += 1
            append_jsonl(
                args.status_jsonl,
                {"sample_id": sample_id, "status": "failed", "error": repr(exc)},
            )
            torch.cuda.empty_cache()
            continue

        completed += 1
        append_jsonl(
            args.status_jsonl,
            {
                "sample_id": sample_id,
                "status": "complete",
                "seed": int(row["seed"]),
                "elapsed_sec": round(time.time() - started, 3),
                "output": str(resized_path),
                "probe": probe,
            },
        )
        atomic_json(
            args.state_json,
            {
                "state": "running",
                "completed": completed,
                "total": len(rows),
                "current": sample_id,
                "failed": failed,
            },
        )

    atomic_json(
        args.state_json,
        {
            "state": "failed" if failed else "complete",
            "completed": completed,
            "total": len(rows),
            "failed": failed,
        },
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
