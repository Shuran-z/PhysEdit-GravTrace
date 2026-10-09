import os
import sys
import json
import argparse
import subprocess
import traceback
from pathlib import Path

import torch
from PIL import Image
import numpy as np
from peft import PeftModel

current_dir = os.path.dirname(os.path.abspath(__file__))
wan_path = os.path.abspath(os.path.join(current_dir, "..", "Wan2.2"))
if wan_path not in sys.path:
    sys.path.insert(0, wan_path)

from wan.configs import WAN_CONFIGS
from wan.textimage2video import WanTI2V

WAN_CONFIG = WAN_CONFIGS['ti2v-5B']


def save_video_ffmpeg(video_tensor, output_path, fps=24):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    video_data = video_tensor.permute(1, 2, 3, 0).cpu().numpy().clip(-1, 1)
    video_data = ((video_data + 1) * 127.5).astype(np.uint8)
    f, h, w, c = video_data.shape
    proc = subprocess.Popen(
        ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
         '-s', f'{w}x{h}', '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
         str(output_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(video_data.tobytes())
    proc.stdin.close()
    proc.wait()


def generate(args):
    torch.cuda.set_device(args.gpu_id)

    # Load engine
    print(f"Loading Wan TI2V engine: {args.model_path}")
    engine = WanTI2V(
        config=WAN_CONFIG,
        checkpoint_dir=args.model_path,
        device_id=args.gpu_id,
        rank=0,
        t5_cpu=args.t5_cpu,
    )

    # Mount LoRA (optional)
    if args.lora_path:
        if not Path(args.lora_path).exists():
            print(f"LoRA path not found: {args.lora_path}, using base model")
        else:
            print(f"Mounting LoRA: {args.lora_path} (weight={args.lora_weight})")
            engine.model = PeftModel.from_pretrained(
                engine.model, args.lora_path,
                adapter_name="default",
                torch_dtype=torch.bfloat16,
            )
            # Scale the LoRA contribution before merging (delta_W *= lora_weight)
            if args.lora_weight != 1.0:
                for module in engine.model.modules():
                    if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                        for adapter in module.scaling:
                            module.scaling[adapter] *= args.lora_weight
            engine.model.merge_and_unload()
            print("LoRA merged.")

    engine.model.eval()
    # public211: bf16 DiT after the fp32 LoRA merge (same as official convert_model_dtype); the fp32 DiT (20 GB)
    # no longer fits a 3090 that other users already occupy with ~2 GB
    engine.model.to(engine.param_dtype)
    torch.cuda.empty_cache()

    # Load prompts
    with open(args.prompt_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        tasks = list(raw_data.items())  # [(key, {text_prompt, image_prompt, ...}), ...]
    elif isinstance(raw_data, list):
        tasks = [(item.get('group_id', i), item) for i, item in enumerate(raw_data)]
    else:
        print("Unsupported JSON format"); return

    if args.num_prompts:
        tasks = tasks[:args.num_prompts]

    print(f"Generating {len(tasks)} prompts, seed={args.seed}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for idx, (group_id, item) in enumerate(tasks):
        group_id = str(group_id).replace('/', '_')
        text_prompt = item.get('text_prompt', item.get('prompt', '')).strip()
        image_path = item.get('image_prompt', item.get('image_path', ''))

        if not text_prompt or not image_path:
            continue

        if not Path(image_path).exists() and args.base_dir:
            image_path = str(Path(args.base_dir) / image_path)
        if not Path(image_path).exists():
            print(f"[{idx+1}/{len(tasks)}] Image not found: {image_path}, skipping")
            continue

        out_dir = output_root / group_id
        out_dir.mkdir(parents=True, exist_ok=True)
        task_seed = int(item.get("seed", args.seed))
        video_path = out_dir / f"seed_{task_seed}.mp4"

        if video_path.exists():
            print(f"[{idx+1}/{len(tasks)}] Skip existing: {group_id}")
            continue

        print(f"[{idx+1}/{len(tasks)}] Generating: {group_id}, seed={task_seed}")
        try:
            image = Image.open(image_path).convert("RGB")
            video_tensor = engine.generate(
                input_prompt=text_prompt,
                img=image,
                max_area=args.max_area,
                frame_num=args.frame_num,
                shift=args.shift,
                sampling_steps=args.sampling_steps,
                guide_scale=args.guide_scale,
                seed=task_seed,
                offload_model=args.offload_model,
            )
            save_video_ffmpeg(video_tensor, video_path, fps=args.fps)
        except Exception as e:
            print(f"  Failed: {e}")
            traceback.print_exc()
        torch.cuda.empty_cache()

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V generation")
    parser.add_argument("--model_path", type=str, required=True, help="Wan2.2-TI2V-5B model path")
    parser.add_argument("--prompt_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--lora_weight", type=float, default=0.2, help="LoRA strength relative to trained scaling (1.0 = full; default 0.2)")
    parser.add_argument("--base_dir", type=str, default=None, help="Base dir for relative image paths")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--max_area", type=int, default=704 * 1280)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
