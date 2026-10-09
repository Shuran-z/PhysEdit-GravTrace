#!/usr/bin/env python3
"""Resumable MiniMax-H3 FL2VA generation (two GPUs; one with a shared offload budget) for a physedit manifest."""

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import av
import imageio_ffmpeg
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video


MODEL = "/XYAIFS00/HDD_POOL/sysu_xdliang/sysu_xdliang_3/zhangshuran/models/MiniMax-H3-FL2VA-diffusers-20260903"
H3_WIDTH = 896
H3_HEIGHT = 512


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: pathlib.Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def probe(path: pathlib.Path, ffmpeg: str):
    if not path.is_file() or path.stat().st_size < 1024:
        return False, "missing_or_too_small"
    result = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    return result.returncode == 0, result.stderr[-1000:]


def video_metadata(path: pathlib.Path):
    container = av.open(str(path))
    streams = [
        {
            "type": stream.type,
            "codec": stream.codec_context.name,
            "width": getattr(stream.codec_context, "width", None),
            "height": getattr(stream.codec_context, "height", None),
            "frames_declared": stream.frames,
            "rate": str(getattr(stream, "average_rate", None)),
        }
        for stream in container.streams
    ]
    frames = sum(isinstance(frame, av.VideoFrame) for frame in container.decode(video=0))
    container.close()
    return frames, streams


def output_row(source_row: dict, output: pathlib.Path, resized: pathlib.Path):
    source_id = source_row["source_sample_id"]
    granularity = source_row["prompt_granularity"]
    return {
        "sample_id": f"minimax_h3_fl2va_{source_id}_{granularity}",
        "source_sample_id": source_id,
        "index": source_row["index"],
        "mode": "i2v",
        "model_name": "minimax_h3_fl2va",
        "conditioning_track": "I2V_1F",
        "condition_policy": "I2V_1F uses only the declared condition image",
        "condition_image": source_row["condition_image"],
        "prompt_file": source_row["prompt_file"],
        "negative_prompt_file": source_row.get("negative_prompt_file"),
        "prompt_field": source_row.get("prompt_field"),
        "prompt_granularity": granularity,
        "scenario": source_row.get("scenario"),
        "template": source_row.get("template"),
        "seed": source_row["seed"],
        "source_width": source_row["source_width"],
        "source_height": source_row["source_height"],
        "generated_video": str(output),
        "generated_video_resized_to_source": str(resized),
        "output_fps": 24.0,
        "num_frames": 124,
        "sample_steps": 50,
        "h3_resolution": {"width": H3_WIDTH, "height": H3_HEIGHT},
        "h3_audio_generation": True,
        "negative_prompt_used": False,
        "gt_join_stage": "score_only",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    model = args.model

    out_root = pathlib.Path(args.out_root)
    runtime = out_root / "runtime"
    state_path = runtime / "generation_state.json"
    status_path = runtime / "generation_status.jsonl"
    canonical_path = runtime / "full_manifest.jsonl"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source_rows = [json.loads(line) for line in pathlib.Path(args.source_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples:
        source_rows = source_rows[: args.max_samples]
    if len({row["sample_id"] for row in source_rows}) != len(source_rows):
        raise RuntimeError("source manifest contains duplicate sample_id values")

    generated_rows = []
    for source in source_rows:
        if source.get("generated_video_resized_to_source"):  # physedit manifests name the outputs
            output, resized = pathlib.Path(source["generated_video"]), pathlib.Path(source["generated_video_resized_to_source"])
        else:
            destination = out_root / "full" / "generated" / source["source_sample_id"] / source["prompt_granularity"]
            output, resized = destination / "generated_video.mp4", destination / "generated_video_resized_to_source.mp4"
        generated_rows.append(output_row(source, output, resized))
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated_rows), encoding="utf-8")

    base = ModularPipeline.from_pretrained(model, local_files_only=True)
    workflow = base.blocks.get_workflow("fl2va")
    prep_block = workflow.sub_blocks.pop("before_encode")
    text_block = workflow.sub_blocks.pop("text_encoder")

    prep = prep_block.init_pipeline(model)
    prep.load_components()
    text_manager = ComponentsManager()
    text_manager.enable_auto_cpu_offload(device=("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"), memory_reserve_margin="4GB")
    conditioner = text_block.init_pipeline(model, components_manager=text_manager)
    conditioner.load_components(dtype=torch.bfloat16)
    generator_manager = text_manager if torch.cuda.device_count() == 1 else ComponentsManager()  # public211: one GPU -> one shared offload budget
    # The smaller 896x512 canvas provides activation headroom without extra transfer stalls.
    generator_manager.enable_auto_cpu_offload(device="cuda:0", memory_reserve_margin="8GB")
    generator = workflow.init_pipeline(model, components_manager=generator_manager)
    generator.load_components(dtype=torch.bfloat16)
    generator_manager.enable_auto_cpu_offload(device="cuda:0", memory_reserve_margin="8GB")  # public211: re-apply the 8GB margin to every component

    completed = 0
    failed = 0
    started_all = time.time()
    for source, row in zip(source_rows, generated_rows):
        output = pathlib.Path(row["generated_video"])
        resized = pathlib.Path(row["generated_video_resized_to_source"])
        output_ok, _ = probe(output, ffmpeg)
        resized_ok, _ = probe(resized, ffmpeg)
        if output_ok and resized_ok:
            completed += 1
            continue

        started = time.time()
        write_json(state_path, {"state": "running", "completed": completed, "failed": failed, "total": len(generated_rows), "current": row["sample_id"], "updated_at": now()})
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_output = output.with_name(output.stem + ".incomplete.mp4")
        temp_resized = resized.with_name(resized.stem + ".incomplete.mp4")
        error = None
        try:
            prompt = pathlib.Path(source["prompt_file"]).read_text(encoding="utf-8").strip()
            image = load_image(source["condition_image"])
            with torch.inference_mode():
                state = prep(image=image, height=int(source.get("gen_height") or H3_HEIGHT), width=int(source.get("gen_width") or H3_WIDTH))  # public211: per-row canvas keeps aspect
                state = conditioner(state=state, prompt=prompt)
                result = generator(
                    state=state,
                    num_frames=124,
                    num_inference_steps=args.steps,
                    generator=torch.Generator().manual_seed(int(source["seed"])),
                    output=["videos", "audio", "sampling_rate"],
                )
            encode_video(result["videos"][0], fps=24, output_path=str(temp_output), audio=result["audio"][0], audio_sample_rate=result["sampling_rate"])
            ok, detail = probe(temp_output, ffmpeg)
            if not ok:
                raise RuntimeError(f"primary decode failed: {detail}")
            frames, streams = video_metadata(temp_output)
            if frames != 124 or not any(stream["type"] == "audio" for stream in streams):
                raise RuntimeError(f"media contract failed: frames={frames}, streams={streams}")
            subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(temp_output), "-vf", f"scale={source['source_width']}:{source['source_height']}", "-r", "24", "-an", str(temp_resized)], check=True)
            ok, detail = probe(temp_resized, ffmpeg)
            if not ok:
                raise RuntimeError(f"resized decode failed: {detail}")
            os.replace(temp_output, output)
            os.replace(temp_resized, resized)
            completed += 1
            event = "complete"
        except Exception as exc:
            failed += 1
            event = "failed"
            error = repr(exc)
            temp_output.unlink(missing_ok=True)
            temp_resized.unlink(missing_ok=True)
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": now(), "sample_id": row["sample_id"], "state": event, "elapsed_seconds": round(time.time() - started, 3), "bytes": output.stat().st_size if output.exists() else 0, "error": error, "steps": args.steps, "resolution": "960x544"}, ensure_ascii=False) + "\n")
        if event == "failed":
            write_json(state_path, {"state": "failed", "completed": completed, "failed": failed, "total": len(generated_rows), "current": row["sample_id"], "error": error, "updated_at": now()})
            raise SystemExit(error)
        write_json(state_path, {"state": "running", "completed": completed, "failed": failed, "total": len(generated_rows), "current": row["sample_id"], "updated_at": now()})
        del state, result
        torch.cuda.empty_cache()

    write_json(state_path, {"state": "complete", "completed": completed, "failed": failed, "total": len(generated_rows), "elapsed_seconds": round(time.time() - started_all, 3), "updated_at": now()})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
