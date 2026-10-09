#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import torch
from diffusers.utils import export_to_video, load_image


PIPELINE_CLASS = {
    "cogvideox_i2v": "CogVideoXImageToVideoPipeline",
    "ltx_i2v": "LTXImageToVideoPipeline",
    "hunyuan_i2v": "HunyuanVideoImageToVideoPipeline",
    "hunyuan15_i2v": "HunyuanVideo15ImageToVideoPipeline",
    "wan_i2v": "WanImageToVideoPipeline",
    "wan_ti2v5b": "WanImageToVideoPipeline",
}


DEFAULT_SETTINGS = {
    "cogvideox_i2v": {"height": 480, "width": 768, "steps": 50, "guidance": 6.0},
    "ltx_i2v": {"height": 480, "width": 720, "steps": 30, "guidance": 3.0},
    "hunyuan_i2v": {"height": 480, "width": 720, "steps": 30, "guidance": 6.0},
    "hunyuan15_i2v": {"height": 480, "width": 848, "steps": 12, "guidance": 6.0},
    "wan_i2v": {"height": 480, "width": 720, "steps": 40, "guidance": 3.5},
    "wan_ti2v5b": {"height": 480, "width": 720, "steps": 40, "guidance": 3.5},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--pipeline", choices=sorted(PIPELINE_CLASS), required=True)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--offload-mode", choices=["model", "group", "sequential", "none"], default=None)
    parser.add_argument(
        "--wan-invert-output",
        action="store_true",
        help="Invert decoded Wan RGB frames before export to repair the converted TI2V-5B checkpoint.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.wan_invert_output and not args.pipeline.startswith("wan_"):
        parser.error("--wan-invert-output is only valid with a Wan pipeline")

    if args.pipeline == "wan_ti2v5b":
        import diffusers

        cls = getattr(diffusers, PIPELINE_CLASS[args.pipeline])
        if "image" not in inspect.signature(cls.__call__).parameters:
            parser.error(
                "The installed Diffusers WanPipeline does not accept an image "
                "condition. Use the official Wan2.2 TI2V runner; refusing to "
                "silently run text-to-video from an I2V manifest."
            )

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    manifest = Path(args.generation_manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = rows[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    status_path = manifest.parent / f"{args.pipeline}_generation_status.jsonl"
    failed = completed = skipped = 0
    pipe = None

    with status_path.open("a", encoding="utf-8") as status_handle:
        for row in selected:
            start = time.time()
            out_dir = Path(row["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / f"{args.pipeline}_generate.log"
            raw_path = Path(row["generated_video"])
            resized_path = Path(row["generated_video_resized_to_source"])
            if args.skip_existing and resized_path.exists() and resized_path.stat().st_size > 0:
                skipped += 1
                _write_status(status_handle, row, "skipped_existing", start, log_path, resized_path)
                continue
            if args.dry_run:
                _write_status(status_handle, row, "dry_run", start, log_path, resized_path)
                continue

            try:
                if pipe is None:
                    pipe = _load_pipeline(args.pipeline, args.model_dir, args.torch_dtype, args.offload_mode, args.device_map)
                _generate_one(pipe, args, row, raw_path, log_path)
                _resize_to_source(raw_path, Path(row["source_video"]), resized_path, fps=float(row.get("output_fps") or 16.0), dims=(int(row["source_width"]), int(row["source_height"])))
            except Exception as exc:
                failed += 1
                _write_status(status_handle, row, "failed", start, log_path, resized_path, error=repr(exc))
                _append_log(log_path, f"\n[runner_error] {repr(exc)}\n{traceback.format_exc()}\n")
                continue
            completed += 1
            _write_status(status_handle, row, "completed", start, log_path, resized_path)

    summary = {
        "manifest": str(manifest),
        "pipeline": args.pipeline,
        "model_dir": str(Path(args.model_dir).resolve()),
        "selected": len(selected),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "status_path": str(status_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


def _load_pipeline(pipeline_key: str, model_dir: str, dtype_name: str, offload_mode: str | None, device_map: str | None):
    import diffusers

    cls_name = PIPELINE_CLASS[pipeline_key]
    cls = getattr(diffusers, cls_name)
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    load_kwargs = {"torch_dtype": dtype}
    if device_map:
        load_kwargs["device_map"] = device_map
    pipe = cls.from_pretrained(model_dir, **load_kwargs)
    if pipeline_key == "hunyuan15_i2v" and hasattr(getattr(pipe, "image_encoder", None), "vision_model"):
        # The pipeline only consumes last_hidden_state. Disabling the unused
        # SigLIP pooling head avoids an Accelerate sequential-offload hook gap.
        pipe.image_encoder.vision_model.use_head = False
    if device_map:
        pass
    elif pipeline_key.startswith("wan_"):
        _enable_wan_offload(pipe, offload_mode or "group")
    elif offload_mode == "none":
        pipe.to("cuda")
    elif offload_mode == "sequential" and hasattr(pipe, "enable_sequential_cpu_offload"):
        if pipeline_key == "hunyuan15_i2v":
            pipe.model_cpu_offload_seq = "image_encoder->text_encoder->text_encoder_2->transformer->vae"
        pipe.enable_sequential_cpu_offload()
    elif hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def _enable_wan_offload(pipe, mode: str) -> None:
    if mode == "none":
        pipe.to("cuda")
        return
    if mode == "sequential" and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
        return
    if mode == "group":
        try:
            from diffusers.hooks import apply_group_offloading

            onload_device = torch.device("cuda")
            offload_device = torch.device("cpu")
            for component_name in ("transformer", "transformer_2", "vae"):
                component = getattr(pipe, component_name, None)
                if component is not None and hasattr(component, "enable_group_offload"):
                    component.enable_group_offload(
                        onload_device=onload_device,
                        offload_device=offload_device,
                        offload_type="leaf_level",
                    )
            text_encoder = getattr(pipe, "text_encoder", None)
            if text_encoder is not None:
                apply_group_offloading(text_encoder, onload_device=onload_device, offload_type="leaf_level")
            return
        except Exception as exc:
            print(f"[wan_offload_warning] group offload unavailable: {exc!r}; falling back to sequential", flush=True)
            if hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload()
                return
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")


def _generate_one(pipe, args: argparse.Namespace, row: dict[str, Any], raw_path: Path, log_path: Path) -> None:
    settings = dict(DEFAULT_SETTINGS[args.pipeline])
    height = int(row.get("gen_height") or args.height or settings["height"])  # public211: per-row canvas keeps aspect
    width = int(row.get("gen_width") or args.width or settings["width"])
    steps = args.num_inference_steps or int(settings["steps"])
    guidance = args.guidance_scale if args.guidance_scale is not None else float(settings["guidance"])
    if args.num_frames is not None:
        num_frames = args.num_frames
    elif args.pipeline == "hunyuan15_i2v":
        num_frames = 121
    else:
        num_frames = int(row.get("num_frames") or 81)

    prompt = Path(row["prompt_file"]).read_text(encoding="utf-8").strip()
    negative_prompt = Path(row["negative_prompt_file"]).read_text(encoding="utf-8").strip()
    image = load_image(row["condition_image"])
    if args.pipeline.startswith("wan_"):
        height, width, image = _wan_resize_like_diffusers_example(pipe, image, height * width)
    generator = torch.Generator(device="cuda").manual_seed(int(row.get("seed") or 42))

    call_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": image,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
    }
    accepted = set(inspect.signature(pipe.__call__).parameters)
    filtered = {key: value for key, value in call_kwargs.items() if key in accepted}
    _write_log(
        log_path,
        {
            "pipeline": args.pipeline,
            "model_dir": args.model_dir,
            "sample_id": row.get("sample_id"),
            "condition_image": row.get("condition_image"),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "wan_invert_output": args.wan_invert_output,
            "accepted_call_args": sorted(filtered),
        },
    )
    with torch.inference_mode():
        result = pipe(**filtered)
    frames = _extract_frames(result)
    if not frames:
        raise RuntimeError(f"pipeline returned no frames: {type(result)}")
    if args.wan_invert_output:
        frames = [_invert_frame(frame) for frame in frames]
    frames = [_prepare_frame_for_export(frame) for frame in frames]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    export_fps = 24 if args.pipeline == "hunyuan15_i2v" else int(float(row.get("output_fps") or 16.0))
    export_to_video(frames, str(raw_path), fps=export_fps)


def _wan_resize_like_diffusers_example(pipe, image, max_area: int):
    aspect_ratio = image.height / image.width
    vae_scale = int(getattr(pipe, "vae_scale_factor_spatial", 16) or 16)
    patch_size = getattr(getattr(pipe, "transformer", None), "config", None)
    if patch_size is not None:
        patch = getattr(patch_size, "patch_size", None) or [1, 2, 2]
        spatial_patch = int(patch[1] if isinstance(patch, (list, tuple)) and len(patch) > 1 else 2)
    else:
        spatial_patch = 2
    mod_value = max(1, vae_scale * spatial_patch)
    height = max(mod_value, round(math.sqrt(max_area * aspect_ratio)) // mod_value * mod_value)
    width = max(mod_value, round(math.sqrt(max_area / aspect_ratio)) // mod_value * mod_value)
    return height, width, image.resize((width, height))


def _extract_frames(result: Any):
    for attr in ("frames", "videos"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            return _flatten_frames(value)
    if isinstance(result, (list, tuple)):
        return _flatten_frames(result)
    return []


def _flatten_frames(value: Any):
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, (list, tuple)) and first:
            return list(first)
        return list(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            array = value
            if array.ndim == 5:
                array = array[0]
            if array.ndim == 4 and array.shape[1] in (1, 3, 4):
                array = np.transpose(array, (0, 2, 3, 1))
            if array.ndim != 4:
                return []
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array, 0.0, 1.0)
                array = (array * 255).round().astype(np.uint8)
            elif array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            return [frame for frame in array]
    except Exception:
        return []
    return []


def _invert_frame(frame: Any):
    try:
        from PIL import Image, ImageOps

        if isinstance(frame, Image.Image):
            return ImageOps.invert(frame.convert("RGB"))
    except ImportError:
        pass

    import numpy as np

    array = np.asarray(frame)
    if np.issubdtype(array.dtype, np.floating):
        return 1.0 - np.clip(array, 0.0, 1.0)
    return 255 - np.clip(array, 0, 255).astype(np.uint8)


def _prepare_frame_for_export(frame: Any):
    from PIL import Image

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    import numpy as np

    array = np.asarray(frame)
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255).round().astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _resize_to_source(generated: Path, source: Path, output: Path, fps: float, dims=None) -> None:
    width, height = dims if dims else _video_size(source)  # public211: dims from the manifest
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


def _write_status(handle, row, status, start_time, log_path, resized_path, **extra) -> None:
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - start_time, 3),
        "sample_id": row.get("sample_id"),
        "model_name": row.get("model_name"),
        "source_video": row.get("source_video"),
        "condition_image": row.get("condition_image"),
        "out_dir": row.get("out_dir"),
        "log_path": str(log_path),
        "generated_video_resized_to_source": str(resized_path),
        "status": status,
        **extra,
    }
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _write_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
