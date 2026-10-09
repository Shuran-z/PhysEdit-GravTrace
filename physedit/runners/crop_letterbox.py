"""DynamiCrafter post-step: rows whose condition image was letterboxed to 16:9 get the bars cropped from the
raw 1024x576 output, then are scaled to the source size (the runner's own resize would stretch them)."""
import json, subprocess, sys, pathlib
done = skipped = 0
for line in open(sys.argv[1]):
    r = json.loads(line)
    if "letterbox_box" not in r: continue
    raw, out = pathlib.Path(r["generated_video"]), pathlib.Path(r["final_resized"])
    if out.is_file() and out.stat().st_size > 0: skipped += 1; continue
    if not (raw.is_file() and raw.stat().st_size > 0): continue
    W, H = [int(v) for v in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height", "-of", "csv=p=0:s=x", str(raw)], capture_output=True, text=True, check=True).stdout.strip().split("x")]
    bx, by, bw, bh = r["letterbox_box"]
    cw, ch = int(round(bw * W)) // 2 * 2, int(round(bh * H)) // 2 * 2
    cx, cy = int(round(bx * W)), int(round(by * H))
    tmp = out.with_name(out.stem + ".incomplete.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-vf",
                    "crop=%d:%d:%d:%d,scale=%s:%s:flags=lanczos" % (cw, ch, cx, cy, r["source_width"], r["source_height"]),
                    "-r", r.get("output_fps", "8.0"), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)], check=True)
    tmp.replace(out); done += 1
print(json.dumps({"cropped": done, "already": skipped}))
