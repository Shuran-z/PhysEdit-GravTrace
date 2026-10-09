"""Restrict all methods to the same fully visible, unclipped frames.

Keep the original declaration and frame numbers for depth joins. GravTrace's
time origin is adjusted to the first retained observation, without changing v0.
This controls observations, not the extra geometric/spin information used by
different estimators. No selection by fitted gravity error is performed.
"""
import argparse
import json
from pathlib import Path


def restrict(s):
    obs, w = s["boxes"], s["window"]
    start = w["start_frame"]
    pairs = [(f, b) for f, b in zip(obs["frames"], obs["xyxy"]) if f >= start]
    pairs = pairs[:w.get("max_frames", len(pairs))]
    seen = w.get("edges_visible", {})
    width, height = s["image_size"]
    keep = [(f, b) for f, b in pairs if all(seen.get(str(f), [True] * 4))
            and b[0] > 1 and b[1] > 1 and b[2] < width - 1 and b[3] < height - 1]
    if len(keep) < 4:
        return None
    first = keep[0][0]
    motion = dict(s["motion"])
    t0 = motion.get("t0", [0., 0.])
    delta = (first - pairs[0][0]) / s["fps"]
    motion["t0"] = [v + delta for v in t0]
    return {**s, "motion": motion, "boxes": {"frames": [f for f, _ in keep], "xyxy": [b for _, b in keep]},
            "window": {**w, "max_frames": len(keep), "contact_check": False}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("samples")
    p.add_argument("out")
    a = p.parse_args()
    counts = {"input": 0, "eligible": 0, "too_few_common_frames": 0}
    with open(a.out, "w") as out:
        for line in open(a.samples):
            s = json.loads(line)
            counts["input"] += 1
            common = restrict(s)
            if common is None:
                counts["too_few_common_frames"] += 1
            else:
                counts["eligible"] += 1
                out.write(json.dumps(common) + "\n")
    Path(a.out + ".coverage.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(counts)


if __name__ == "__main__":
    main()
