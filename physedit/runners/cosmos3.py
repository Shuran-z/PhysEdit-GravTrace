#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import time

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def probe(path, ffmpeg):
    path = pathlib.Path(path)
    if not path.is_file() or path.stat().st_size < 1024:
        return False, "missing_or_too_small"
    result = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    return result.returncode == 0, result.stderr[-2000:]


def prompt_json(text, width, height, frames, fps):
    seconds = frames / fps
    return {
        "subjects": [],
        "background_setting": "",
        "lighting": {},
        "aesthetics": {},
        "cinematography": {},
        "style_medium": "",
        "artistic_style": "",
        "context": text,
        "actions": [{"time": f"0:00-0:{seconds:05.2f}", "description": text}],
        "text_and_signage_elements": [],
        "segments": [{"segment_index": 0, "time_range": f"0:00-0:{seconds:05.2f}", "description": text, "key_changes": text, "camera": ""}],
        "transitions": [],
        "temporal_caption": text,
        "audio_description": "",
        "resolution": {"W": width, "H": height},
        "aspect_ratio": f"{width},{height}",
        "duration": f"{seconds:.4f}s",
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--status-jsonl", required=True)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--ffmpeg", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in pathlib.Path(args.manifest).read_text().splitlines() if line.strip()]
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        enable_safety_checker=False,
        local_files_only=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)

    completed = 0
    failed = 0
    status_path = pathlib.Path(args.status_jsonl)
    for row in rows:
        output = pathlib.Path(row["generated_video"])
        resized = pathlib.Path(row["generated_video_resized_to_source"])
        output_ok, _ = probe(output, args.ffmpeg)
        resized_ok, _ = probe(resized, args.ffmpeg)
        if output_ok and resized_ok:
            completed += 1
            continue

        started = time.time()
        atomic_json(args.state_json, {"state": "running", "node": args.node, "completed": completed, "failed": failed, "total": len(rows), "current": row["sample_id"], "safety_checker": False, "conditioning_track": "I2V_1F", "updated_at": utc_now()})
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp_output = output.with_name(output.stem + ".incomplete.mp4")
        tmp_resized = resized.with_name(resized.stem + ".incomplete.mp4")
        try:
            prompt = pathlib.Path(row["prompt_file"]).read_text().strip()
            negative = pathlib.Path(row["negative_prompt_file"]).read_text().strip()
            width, height = int(row.get("gen_width") or 832), int(row.get("gen_height") or 480)  # public211: per-row canvas keeps aspect
            frames = int(row.get("num_frames", 81))
            fps = float(row.get("output_fps", 16.0))
            result = pipe(
                prompt=json.dumps(prompt_json(prompt, width, height, frames, fps), ensure_ascii=False),
                negative_prompt=json.dumps({"temporal_caption": negative}, ensure_ascii=False),
                image=load_image(row["condition_image"]),
                num_frames=frames,
                height=height,
                width=width,
                fps=fps,
                num_inference_steps=int(row.get("cosmos_num_inference_steps", 35)),
                guidance_scale=float(row.get("cosmos_guidance_scale", 6.0)),
                generator=torch.Generator(device="cuda").manual_seed(int(row["seed"])),
                add_resolution_template=False,
                add_duration_template=False,
                enable_safety_check=False,
            )
            export_to_video(result.video, str(tmp_output), fps=fps)
            ok, detail = probe(tmp_output, args.ffmpeg)
            if not ok:
                raise RuntimeError(f"generated decode failed: {detail}")
            subprocess.run(
                [args.ffmpeg, "-y", "-v", "error", "-i", str(tmp_output), "-vf", f"scale={row['source_width']}:{row['source_height']}", "-r", str(fps), "-an", str(tmp_resized)],
                check=True,
            )
            ok, detail = probe(tmp_resized, args.ffmpeg)
            if not ok:
                raise RuntimeError(f"resized decode failed: {detail}")
            os.replace(tmp_output, output)
            os.replace(tmp_resized, resized)
            completed += 1
            event_state = "complete"
            error = None
        except Exception as exc:
            failed += 1
            event_state = "failed"
            error = repr(exc)
            tmp_output.unlink(missing_ok=True)
            tmp_resized.unlink(missing_ok=True)

        with status_path.open("a") as handle:
            handle.write(json.dumps({"time": utc_now(), "node": args.node, "sample_id": row["sample_id"], "state": event_state, "elapsed_seconds": round(time.time() - started, 3), "bytes": output.stat().st_size if output.exists() else 0, "error": error, "safety_checker": False, "conditioning_track": "I2V_1F"}, ensure_ascii=False) + "\n")
        atomic_json(args.state_json, {"state": "running" if event_state == "complete" else "failed", "node": args.node, "completed": completed, "failed": failed, "total": len(rows), "current": row["sample_id"], "safety_checker": False, "conditioning_track": "I2V_1F", "updated_at": utc_now()})
        if event_state == "failed":
            raise SystemExit(error)

    atomic_json(args.state_json, {"state": "complete", "node": args.node, "completed": completed, "failed": failed, "total": len(rows), "current": None, "safety_checker": False, "conditioning_track": "I2V_1F", "updated_at": utc_now()})


if __name__ == "__main__":
    main()
