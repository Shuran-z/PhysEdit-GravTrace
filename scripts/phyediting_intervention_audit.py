"""Inventory named interventions without promoting filenames into physical truth.

Uses only local manifests. Does not download media, change splits, or launch jobs.
Run: python scripts/phyediting_intervention_audit.py --out docs/intervention_audit
All complete triples remain candidates until configuration and outcome validation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

BRANCH = re.compile(r"(?:^|_)(minus|star|plus)(?=_|$)")
REPEAT = re.compile(r"(?:^|_)rep(\d+)(?=_|$)")
SERVER_REPEAT = re.compile(r"(?:^|__)r(\d+)$")
SPLITS = ("train", "val", "test_id", "test_ood_template")
LEVELS = ("minus", "star", "plus")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_label(row):
    """Read explicit branch/repeat tokens only; never infer outcome or parameters."""
    server = row.get("source_kind") == "ssh"
    text = row.get("source_slot", "") if server else row["sample_id"]
    matches = list(BRANCH.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    repeat = (SERVER_REPEAT if server else REPEAT).search(text)
    # Existing source keys deliberately bind background and repeated variants.
    # They are a family label, not proof that only one physical parameter changed.
    family = row.get("source_group_key")
    return {
        "branch": match.group(1),
        "repeat": int(repeat.group(1)) if repeat else None,
        "family": family,
        "sample_prefix": text.split("__", 1)[0],
    }


def reference_present(row):
    return bool(row.get("sidecar_paths") or row.get("config_path"))


def group_key(row, label):
    # Never fill a missing branch using another revision, release, gravity,
    # background, repeat, or sample-version prefix.
    return (
        row["engine"], row.get("source_kind"), row.get("repo_id", row.get("host")),
        row.get("revision"), row.get("release"), label["sample_prefix"],
        row["event_id"], row.get("background_id"), row.get("gravity_id"),
        label["family"], label["repeat"],
    )


def audit(rows, exposed_groups):
    buckets = defaultdict(list)
    counters = defaultdict(Counter)
    missing_repeat = []
    for row in rows:
        key = f"{row['engine']}/{row['event_id']}"
        counters[key]["trajectories"] += 1
        counters[key]["videos"] += len(row["videos"])
        label = parse_label(row)
        if label is None:
            counters[key]["no_unambiguous_branch_token"] += 1
            continue
        counters[key]["named_branch_trajectories"] += 1
        if label["repeat"] is None or not label["family"]:
            counters[key]["branch_without_explicit_repeat_or_family"] += 1
            missing_repeat.append(row["record_id"])
            continue
        buckets[group_key(row, label)].append((row, label))

    groups = []
    for key, members in sorted(buckets.items(), key=lambda kv: repr(kv[0])):
        counts = Counter(label["branch"] for _, label in members)
        complete = set(counts) == set(LEVELS) and all(n == 1 for n in counts.values())
        splits = sorted({r["split"] for r, _ in members})
        leakage_ids = sorted({r["leakage_group_id"] for r, _ in members})
        views_complete = all(set(r["videos"]) == {"cam01", "cam02", "cam03"} for r, _ in members)
        references = all(reference_present(r) for r, _ in members)
        reasons = []
        if set(counts) != set(LEVELS):
            reasons.append("missing_branch")
        if any(n != 1 for n in counts.values()):
            reasons.append("ambiguous_duplicate_branch")
        if len(splits) != 1:
            reasons.append("cross_split")
        if len(leakage_ids) != 1:
            reasons.append("multiple_leakage_groups")
        if not views_complete:
            reasons.append("incomplete_views")
        if not references:
            reasons.append("missing_metadata_reference")
        same_dev_group = bool(set(leakage_ids) & exposed_groups)
        representative = members[0][0]
        family_key = key[:7] + (representative.get("gravity_id"), members[0][1]["family"])
        groups.append({
            "candidate_id": digest(key), "family_id": digest(family_key),
            "engine": representative["engine"], "event_id": representative["event_id"],
            "source_kind": representative.get("source_kind"),
            "source_identity": {k: representative.get(k) for k in ("repo_id", "revision", "host")},
            "release": representative.get("release"),
            "source_group_key": members[0][1]["family"],
            "background_id": representative.get("background_id"),
            "gravity_id": representative.get("gravity_id"),
            "repeat": members[0][1]["repeat"],
            "splits": splits, "leakage_group_ids": leakage_ids,
            "branch_counts": dict(counts), "complete_named_triple": complete,
            "metadata_references_present": references,
            "structural_review_candidate": not reasons,
            "review_blockers": reasons,
            "known_metric_development_group": same_dev_group,
            "unseen_status": "not_certified_by_this_audit",
            "physical_validation": "pending_config_runtime_outcome_and_visual_checks",
            "qa_statuses": sorted({r.get("qa_status", "unknown") for r, _ in members}),
            "release_statuses": sorted({r.get("release_status", "unknown") for r, _ in members}),
            "members": [{"branch": label["branch"], "record_id": r["record_id"],
                         "sample_id": r["sample_id"], "split": r["split"],
                         "sidecar_reference_count": len(r.get("sidecar_paths", [])),
                         "config_reference_present": bool(r.get("config_path"))}
                        for r, label in sorted(members, key=lambda m: (m[1]["branch"], m[0]["record_id"]))],
        })
    for key in counters:
        relevant = [g for g in groups if f"{g['engine']}/{g['event_id']}" == key]
        complete = [g for g in relevant if g["complete_named_triple"]]
        candidates = [g for g in relevant if g["structural_review_candidate"]]
        counters[key].update({
            "candidate_buckets": len(relevant), "complete_named_triples": len(complete),
            "complete_versioned_families": len({g["family_id"] for g in complete}),
            "complete_leakage_groups": len({x for g in complete for x in g["leakage_group_ids"]}),
            "structural_review_candidates": len(candidates),
            "train_review_candidates": sum(g["splits"] == ["train"] for g in candidates),
            "known_metric_dev_complete_triples": sum(g["known_metric_development_group"] for g in complete),
        })
        for split in SPLITS:
            counters[key][f"complete_triples_{split}"] = sum(g["splits"] == [split] for g in complete)
    summary = {
        "scope": "Named branch inventory only; file references not media or physical validation",
        "input_trajectories": len(rows), "input_videos": sum(len(r["videos"]) for r in rows),
        "by_engine_event": {k: dict(v) for k, v in sorted(counters.items())},
        "complete_named_triples": sum(g["complete_named_triple"] for g in groups),
        "structural_review_candidates": sum(g["structural_review_candidate"] for g in groups),
        "physics_certification": "not_performed",
        "unresolved_explicit_branch_records": missing_repeat,
        "limitations": [
            "Filename branches do not establish single-variable interventions, numeric ranges, or distinct outcomes.",
            "Backgrounds, repeats and views do not establish independent physical settings.",
            "No-match to known development groups does not certify unseen data.",
            "Only explicit minus/star/plus and repeat tokens are paired; other designs remain unclassified.",
            "Version-strict grouping may undercount legitimate cross-release families; no automatic merging.",
            "sidecar_paths or config_path indicates a reference, not verified labels or runtime values.",
        ],
    }
    return summary, groups


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-root", type=Path, default=repo.parent / "dataset_splits_v2")
    ap.add_argument("--development-audit", type=Path, default=repo / "docs/CVPR_PROGRESS_AUDIT_2026-10-09.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    development = json.loads(args.development_audit.read_text(encoding="utf-8"))
    exposed = {r["leakage_group_id"] for r in development["development_split_join"]["matches"]}
    args.out.mkdir(parents=True, exist_ok=True)
    for scope in ("huggingface", "with_isaac_server"):
        paths = [args.splits_root / scope / engine / (split + ".jsonl")
                 for engine in ("genesis", "isaac") for split in SPLITS]
        rows = [r for path in paths for r in read_jsonl(path)]
        if len({r["record_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate record IDs in input scope")
        summary, groups = audit(rows, exposed)
        summary["scope_name"] = scope
        summary["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in paths + [args.development_audit]}
        # Metadata-review seed only, on train: one representative per event/gravity.
        pilots = {}
        for g in sorted(groups, key=lambda g: g["candidate_id"]):
            if g["structural_review_candidate"] and g["splits"] == ["train"]:
                pilots.setdefault((g["engine"], g["event_id"], g["gravity_id"]), g)
        summary["train_metadata_review_groups"] = len(pilots)
        (args.out / f"{scope}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_jsonl(args.out / f"{scope}_groups.jsonl", groups)
        by_record = {r["record_id"]: r for r in rows}
        review_rows = []
        for group in pilots.values():
            review_members = []
            for member in group["members"]:
                row = by_record[member["record_id"]]
                review_members.append({**member, "cam01": row["videos"]["cam01"],
                                       "sidecar_paths": row.get("sidecar_paths", []),
                                       "config_path": row.get("config_path")})
            review_rows.append({**group, "members": review_members})
        write_jsonl(args.out / f"{scope}_train_review.jsonl", review_rows)
        print(scope, "complete named triples", summary["complete_named_triples"],
              "structural candidates", summary["structural_review_candidates"],
              "train metadata review", len(pilots))


if __name__ == "__main__":
    main()
