"""Camera motion in generated videos, as a similarity transform of every frame onto the first one.

Generated clips often pan or zoom although the reference camera is fixed. The object's box then moves with the
camera, which a fit would take for object motion. This estimates, per frame, the 2D similarity (scale, rotation,
shift) that maps the frame onto frame 0 from ORB features on the static scene (RANSAC drops the moving objects),
so that boxes can be brought back into the condition frame's image before fitting.

    python scripts/camera_motion.py JOBS.jsonl OUT.jsonl     (JOBS: SAM2 job lines with "id" and "video")

Output per video: {"id", "frames": [...], "affine": [[a, b, tx], [c, d, ty]] per frame (frame -> frame 0),
"inliers": [...], "shift_px": [...]} where shift_px is how far the image centre moves.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

MIN_INLIERS = 25


def estimate(video: str, max_side: int = 640) -> dict:
    cap = cv2.VideoCapture(video)
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ref = None
    out = {"frames": [], "affine": [], "inliers": [], "shift_px": []}
    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        gray = cv2.cvtColor(cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if ref is None:
            ref = (kp, des, s, w, h)
            m, n_in = np.array([[1.0, 0, 0], [0, 1.0, 0]]), len(kp)
        else:
            m, n_in = np.array([[1.0, 0, 0], [0, 1.0, 0]]), 0
            if des is not None and ref[1] is not None and len(kp) >= MIN_INLIERS:
                matches = matcher.match(des, ref[1])
                if len(matches) >= MIN_INLIERS:
                    src = np.float32([kp[x.queryIdx].pt for x in matches]) / s
                    dst = np.float32([ref[0][x.trainIdx].pt for x in matches]) / ref[2]
                    est, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                                           maxIters=4000, confidence=0.999)
                    if est is not None and int(inl.sum()) >= MIN_INLIERS:
                        m, n_in = est, int(inl.sum())
        centre = np.array([w / 2, h / 2, 1.0])
        out["frames"].append(k)
        out["affine"].append(np.round(m, 6).tolist())
        out["inliers"].append(n_in)
        out["shift_px"].append(round(float(np.linalg.norm(m @ centre - centre[:2])), 2))
        k += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    args = ap.parse_args()
    with open(args.out, "w") as f:
        for line in open(args.jobs):
            job = json.loads(line)
            try:
                rec = {"id": job["id"], **estimate(job["video"])}
            except Exception as exc:  # unreadable video
                rec = {"id": job["id"], "error": repr(exc)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if "shift_px" in rec:
                print(job["id"][:60], "max shift", max(rec["shift_px"]), "min inliers", min(rec["inliers"][1:] or [0]), flush=True)


if __name__ == "__main__":
    main()
