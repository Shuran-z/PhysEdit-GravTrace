"""Join predictions with truth. This is the only module that reads `truth`."""
from __future__ import annotations

import numpy as np

PENALTY_CAP = 1.0  # a missing or failed fit counts as 100 % error


def rows(samples: list[dict], preds: dict[str, dict]) -> list[dict]:
    out = []
    for s in samples:
        p = preds.get(s["id"], {})
        truth = float(s["truth"]["gravity"])
        g = p.get("gravity") if p.get("status") in ("ok", "at_bound") else None
        err = None if g is None else abs(g - truth) / truth
        out.append({"id": s["id"], "source": s.get("source", ""), "scenario": s["scenario"], "status": p.get("status", "missing"),
                    "gravity_true": truth, "gravity_pred": g, "rel_error": err,
                    "penalty": PENALTY_CAP if err is None else min(err, PENALTY_CAP)})
    return out


def summarize(scored: list[dict]) -> dict:
    """Error statistics over fitted rows plus the coverage-penalised mean over all rows."""
    errors = np.array([r["rel_error"] for r in scored if r["rel_error"] is not None])
    stats = {"n": len(scored), "fitted": int(errors.size), "penalised_mean": float(np.mean([r["penalty"] for r in scored]))}
    if errors.size:
        stats.update(mean=float(errors.mean()), median=float(np.median(errors)), max=float(errors.max()),
                     within_30pct=int((errors <= 0.3).sum()))
    return stats


def summary(scored: list[dict]) -> dict:
    groups = {"all": scored}
    for key in ("source", "scenario"):
        for value in sorted({r[key] for r in scored}):
            groups[f"{key}={value}"] = [r for r in scored if r[key] == value]
    return {name: summarize(group) for name, group in groups.items()}
