"""Build one model's generation manifests on the site that runs it (standard library; Pillow only for letterboxing).

usage: python build.py SPEC.json      (the spec is written by the driver)

Every row comes from the benchmark's template row (same source, condition frame, prompt and seed); the model sets
its frame count, frame rate, canvas and output paths. Writes {dir}/manifest.jsonl (all rows) and one manifest per
worker: manifest_w0.jsonl walks the rows forwards and manifest_w1.jsonl backwards, so two workers meet in the middle;
with `split` (runners that process their whole manifest as one batch) the workers get disjoint halves instead. When
the model needs them: per-orientation manifests, VideoGPA prompt files and 16:9 letterboxed condition images.
"""
FORMAT = 2              # bump when the files written here change: runs built by an older format are rebuilt
import collections
import json
import os
import sys


def orientation(r):
    w, h = int(r["source_width"]), int(r["source_height"])
    return "square" if abs(w - h) <= 0.05 * max(w, h) else ("landscape" if w > h else "portrait")


def interleave(rows):
    """Round-robin over source benchmarks, coarse and fine of a source adjacent, so a partial run covers all of them."""
    by = collections.OrderedDict()
    for r in rows:
        by.setdefault(r["benchmark"], collections.OrderedDict()).setdefault(r["source_sample_id"], []).append(r)
    queues, out = [list(v.values()) for v in by.values()], []
    while any(queues):
        for q in queues:
            if q:
                out += sorted(q.pop(0), key=lambda r: r["prompt_granularity"])
    return out


def letterbox(src, dst_dir):
    """Pad a condition image with black bars to 16:9 (DynamiCrafter-1024 has a fixed 1024x576 canvas)."""
    from PIL import Image
    im = Image.open(src).convert("RGB")
    w, h = im.size
    nw, nh = max(w, -(-h * 16 // 9)), max(h, -(-w * 9 // 16))
    nw, nh = (nw, -(-nw * 9 // 16)) if nw * 9 >= nh * 16 else (-(-nh * 16 // 9), nh)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    dst = os.path.join(dst_dir, os.path.basename(src))
    if not os.path.exists(dst):
        canvas = Image.new("RGB", (nw, nh))
        canvas.paste(im, (x0, y0))
        canvas.save(dst)
    return dst, [x0 / nw, y0 / nh, w / nw, h / nh]


def dump(path, rows):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    os.replace(path + ".tmp", path)


def build(spec):
    d, run = spec["dir"], spec["run"]
    fields = {k: v for k, v in spec.get("fields", {}).items() if not isinstance(v, dict)}
    fields.update(spec.get("fields", {}).get(run, {}))
    rows = interleave([json.loads(l) for l in open(spec["template"], encoding="utf-8") if l.strip()])
    out = []
    for r in rows[: spec.get("limit") or None]:
        for old, new in spec.get("remap", []):
            r = {k: v.replace(old, new) if isinstance(v, str) else v for k, v in r.items()}
        r = {k: v for k, v in r.items() if k not in ("gen_width", "gen_height", "final_resized", "letterbox_box")}
        sid = "%s_%s_%s" % (spec["model"], r["source_sample_id"], r["prompt_granularity"])
        o = "%s/generated/%s" % (d, sid)
        r.update(sample_id=sid, model_name=spec["name"], pipeline=spec["model"], model_dir="", run=run,
                 num_frames=spec["frames"], output_fps=float(spec["fps"]), orientation=orientation(r), out_dir=o,
                 generated_video=o + "/generated_video.mp4", generated_video_resized_to_source=o + "/resized.mp4")
        if spec.get("canvas"):
            r["gen_width"], r["gen_height"] = spec["canvas"][r["orientation"]]
        r.update({k: v.format(inputs=spec["inputs"], **r) if isinstance(v, str) else v for k, v in fields.items()})
        if spec.get("letterbox") and r["orientation"] != "landscape":
            os.makedirs(d + "/conditioning_letterbox", exist_ok=True)
            r["condition_image_original"] = r["condition_image"]
            r["condition_image"], r["letterbox_box"] = letterbox(r["condition_image"], d + "/conditioning_letterbox")
            r["generated_video_resized_to_source"] = o + "/runner_stretched_unused.mp4"   # crop_letterbox.py makes resized.mp4
            r["final_resized"] = o + "/resized.mp4"
        need = ["condition_image", "prompt_file", "negative_prompt_file"] + [k for k in fields if k.endswith("_video")]
        missing = ["%s=%s" % (k, r[k]) for k in need if not os.path.isfile(str(r[k]))]
        if missing:
            raise SystemExit("%s: missing %s" % (sid, ", ".join(missing)))
        out.append(r)
    os.makedirs(d + "/logs", exist_ok=True)
    dump(d + "/manifest.jsonl", out)
    half = (len(out) + 1) // 2
    for w, rows_ in (("w0", out[:half] if spec.get("split") else out), ("w1", out[half:] if spec.get("split") else out[::-1])):
        dump("%s/manifest_%s.jsonl" % (d, w), rows_)
        if spec.get("split_orientation"):
            for o in ("landscape", "portrait", "square"):
                dump("%s/manifest_%s_%s.jsonl" % (d, w, o), [r for r in rows_ if r["orientation"] == o])
        if spec.get("prompts_json"):     # VideoGPA's script reads {id: {text_prompt, image_prompt}}
            prompts = {r["sample_id"]: {"text_prompt": open(r["prompt_file"], encoding="utf-8").read().strip(),
                                        "image_prompt": r["condition_image"]} for r in rows_}
            with open("%s/prompts_%s.json" % (d, w), "w", encoding="utf-8") as f:
                json.dump(prompts, f, ensure_ascii=False, indent=0)
    return len(out)


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    print(json.dumps({"rows": build(spec), "dir": spec["dir"]}))
