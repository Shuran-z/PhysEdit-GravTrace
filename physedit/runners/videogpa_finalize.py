"""VideoGPA post-step: its script writes <raw_dir>/<sample_id>/seed_<seed>.mp4; place it at the manifest's
generated_video and scale to the source size at the model fps."""
import json, shutil, subprocess, sys, pathlib
manifest, raw_dir, seed = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
done = 0
for line in open(manifest):
    r = json.loads(line)
    src = raw_dir / r["sample_id"] / ("seed_%s.mp4" % seed)
    gen, res = pathlib.Path(r["generated_video"]), pathlib.Path(r["generated_video_resized_to_source"])
    if res.is_file() and res.stat().st_size > 0: continue
    if not (src.is_file() and src.stat().st_size > 0): continue
    gen.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, gen)
    tmp = res.with_name(res.stem + ".incomplete.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(gen), "-vf", "scale=%s:%s:flags=lanczos" % (r["source_width"], r["source_height"]),
                    "-r", r.get("output_fps", "15.0"), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)], check=True)
    tmp.replace(res); done += 1
print(json.dumps({"finalized": done}))
