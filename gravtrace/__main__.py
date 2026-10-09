"""Command line.

  python -m gravtrace fit MANIFEST.jsonl -o PRED.jsonl [--workers N] [--root OLD=NEW]
  python -m gravtrace score MANIFEST.jsonl PRED.jsonl [-o SUMMARY.json]

`fit` removes each sample's `truth` before any worker sees it; `score` joins it back.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

from . import score
from .fit import safe_fit


def load(path: str, root: str | None = None) -> list[dict]:
    samples = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if root:
        old, new = root.split("=", 1)
        for s in samples:
            s["masks"]["dir"] = s["masks"]["dir"].replace(old, new, 1)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(prog="gravtrace")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("manifest")
    fit.add_argument("-o", "--out", required=True)
    fit.add_argument("--workers", type=int, default=1)
    fit.add_argument("--root", help="rewrite mask paths: OLD_PREFIX=NEW_PREFIX")
    sc = sub.add_parser("score")
    sc.add_argument("manifest")
    sc.add_argument("predictions")
    sc.add_argument("-o", "--out")
    args = parser.parse_args()

    if args.command == "fit":
        blind = [{k: v for k, v in s.items() if k != "truth"} for s in load(args.manifest, args.root)]
        with Pool(args.workers) as pool, open(args.out, "w", encoding="utf-8") as out:
            for pred in pool.imap(safe_fit, blind, chunksize=1):
                out.write(json.dumps(pred) + "\n")
    else:
        preds = {p["id"]: p for p in (json.loads(line) for line in open(args.predictions, encoding="utf-8"))}
        result = score.summary(score.rows(load(args.manifest), preds))
        text = json.dumps(result, indent=1)
        print(text)
        if args.out:
            open(args.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
