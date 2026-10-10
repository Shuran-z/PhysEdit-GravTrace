"""Audit the user-confirmed joint-training design against a local inventory.

Produces event-pool routing, NOT train/val/ID assignments or QA certification.
Keeps old v2 splits as provenance only. No media access or remote operations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

POOLS = ("train_pool", "test_ood_1", "test_ood_2")


def validate_plan(plan):
    events = set(plan["events"])
    flat = [e for pool in POOLS for e in plan[pool]]
    if len(flat) != len(set(flat)) or set(flat) != events:
        raise ValueError("Every event must belong to exactly one pool")
    ratios = [Fraction(str(v)) for v in plan["ratios"].values()]
    if any(r <= 0 for r in ratios) or sum(ratios) != 1:
        raise ValueError("Invalid train/val/ID ratios")
    for counts in plan["video_targets"].values():
        if set(counts) != events or any(type(n) is not int or n < 0 or n % 3 for n in counts.values()):
            raise ValueError("Targets must be explicit nonnegative three-view counts")


def target_counts(plan):
    validate_plan(plan)
    table = defaultdict(dict)
    for engine, counts in plan["video_targets"].items():
        pool = sum(counts[e] for e in plan["train_pool"])
        for split, ratio in plan["ratios"].items():
            n = pool * Fraction(str(ratio))
            if n.denominator != 1:
                raise ValueError("Target count is not integral")
            table[split][engine] = int(n)
        for split in POOLS[1:]:
            table[split][engine] = sum(counts[e] for e in plan[split])
    return {s: {**v, "total": sum(v.values())} for s, v in table.items()}


def reference_status(row):
    paths = row.get("sidecar_paths", [])
    return {
        "any_reference": bool(paths or row.get("config_path")),
        "non_config_sidecar_reference": any("/configs/" not in p.replace("\\", "/") for p in paths),
        "trajectory_file_verified": False,
        "note": "Reference presence does not verify complete trajectories, decoding or labels",
    }


def catalog_sample_ids(path):
    """Exact directory IDs or normalized metadata filenames, never substrings."""
    parts = path.split("/")
    stem = re.sub(r"\.(?:jsonl|json|npz)(?:\.gz)?$", "", parts[-1])
    stem = re.sub(r"__(?:cam0[123](?:__.*)?|trajectory)$", "", stem)
    return set(parts[:-1]) | {stem}


def recover_missing_references(rows, catalog):
    """Find pinned catalog sidecars for rows lacking references; do not read media."""
    wanted = defaultdict(list)
    for row in rows:
        if row.get("source_kind") == "huggingface" and not reference_status(row)["any_reference"]:
            wanted[(row["repo_id"], row["sample_id"])].append(row)
    owners = {"phyeditingvideo": "phyeditingvideo/Phyediting", "phy2": "phy2/phydataset"}
    recovered = defaultdict(list)
    sources = {}
    for owner, repo in owners.items():
        pending = [r for (rep, _), rr in wanted.items() if rep == repo for r in rr]
        if not pending:
            continue
        info_path = catalog / f"{owner}_info.json"
        revision = json.loads(info_path.read_text(encoding="utf-8"))["sha"]
        if any(r["revision"] != revision for r in pending):
            raise ValueError("Catalog revision does not match missing-reference inventory")
        sources[str(info_path)] = hashlib.sha256(info_path.read_bytes()).hexdigest()
        for file in sorted(catalog.glob(owner + "__tree__*.json")):
            sources[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            for entry in json.loads(file.read_text(encoding="utf-8")):
                path = entry.get("path", "")
                if entry.get("type") != "file" or not re.search(r"\.(?:jsonl|json|npz)(?:\.gz)?$", path):
                    continue
                for sid in catalog_sample_ids(path):
                    for row in wanted.get((repo, sid), []):
                        if path.startswith(row["release"] + "/"):
                            recovered[row["record_id"]].append(path)
    result = []
    for rr in wanted.values():
        for row in rr:
            paths = sorted(set(recovered[row["record_id"]]))
            result.append({"record_id": row["record_id"], "sample_id": row["sample_id"],
                           "event_id": row["event_id"], "engine": row["engine"],
                           "repo_id": row["repo_id"], "revision": row["revision"],
                           "release": row["release"], "sidecar_candidates": paths,
                           "state_or_metadata_candidates": [p for p in paths if "/metadata/" in p or p.endswith("/state.json")],
                           "content_validation": "not_read_or_verified"})
    return result, sources


def audit(plan, rows, development_records):
    validate_plan(plan)
    pool_for = {e: pool for pool in POOLS for e in plan[pool]}
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate inventory record IDs")
    for row in rows:
        if row["event_id"] not in pool_for or row["engine"] not in plan["video_targets"]:
            raise ValueError("Unknown inventory event or engine")
    # Check group and known hash constraints against new event routing.
    links = defaultdict(set)
    for row in rows:
        pool = pool_for[row["event_id"]]
        links["group:" + row["leakage_group_id"]].add(pool)
        for video in row["videos"].values():
            if video.get("sha256"):
                links["sha256:" + video["sha256"]].add(pool)
    conflicts = {k for k, v in links.items() if len(v) > 1}
    routing = []
    for row in rows:
        keys = {"group:" + row["leakage_group_id"]} | {
            "sha256:" + v["sha256"] for v in row["videos"].values() if v.get("sha256")}
        blockers = sorted(keys & conflicts)
        routing.append({
            "record_id": row["record_id"], "sample_id": row["sample_id"],
            "engine": row["engine"], "event_id": row["event_id"],
            "planned_pool": pool_for[row["event_id"]],
            "route_status": "quarantine_pool_conflict" if blockers else "event_pool_only",
            "assignment_status": "not_a_final_split_or_training_ready_record",
            "previous_v2_split": row.get("split"),
            "leakage_group_id": row["leakage_group_id"],
            "conflict_keys": blockers, "video_references": len(row["videos"]),
            "references": reference_status(row),
            "release_status": row.get("release_status"), "qa_status": row.get("qa_status"),
        })
    inventory = []
    for event in plan["events"]:
        for engine, targets in plan["video_targets"].items():
            rr = [r for r in rows if r["engine"] == engine and r["event_id"] == event]
            actual = sum(len(r["videos"]) for r in rr)
            legacy = sum(len(r["videos"]) for r in rr if r.get("release_status") == "legacy_scan_not_52800_contract")
            inventory.append({
                "event_id": event, "engine": engine, "pool": pool_for[event],
                "target_videos": targets[event], "inventory_videos": actual,
                "inventory_trajectories": len(rr),
                "nominal_shortfall_videos": max(0, targets[event] - actual),
                "nominal_surplus_videos": max(0, actual - targets[event]),
                "legacy_scan_videos": legacy,
                "nonlegacy_inventory_videos_not_QA_certified": actual - legacy,
                "missing_any_metadata_reference_trajectories": sum(not reference_status(r)["any_reference"] for r in rr),
                "without_non_config_sidecar_reference_trajectories": sum(not reference_status(r)["non_config_sidecar_reference"] for r in rr),
                "known_source_groups": len({r["leakage_group_id"] for r in rr}),
                "qa_status_counts": dict(Counter(r.get("qa_status", "unknown") for r in rr)),
                "zero_target_is_intentional": targets[event] == 0,
                "actual_usable_videos": None,
            })
    exposure = Counter()
    for record in development_records:
        match = re.match(r"T\d{2}", record["id"])
        if match and match.group() in pool_for:
            exposure[pool_for[match.group()]] += 1
    summary = {
        "plan_version": plan["version"], "target_counts": target_counts(plan),
        "target_total_videos": sum(sum(c.values()) for c in plan["video_targets"].values()),
        "inventory_total_videos": sum(len(r["videos"]) for r in rows),
        "inventory_pool_videos": {p: {e: sum(x["inventory_videos"] for x in inventory if x["pool"] == p and x["engine"] == e)
                                      for e in plan["video_targets"]} for p in POOLS},
        "cross_pool_conflict_keys": sorted(conflicts),
        "quarantined_records": sum(bool(r["conflict_keys"]) for r in routing),
        "known_metric_development_items_by_new_pool": dict(exposure),
        "event_inventory": inventory,
        "final_train_val_id_assignment": "pending_numeric_intervals_labels_QA_and_group_coverage",
        "trajectory_completeness": "not_verified_from_references",
        "limitations": [
            "Inventory may include legacy, calibration-only and visual-QA-pending files.",
            "Surplus videos in one event do not fill shortages in another event.",
            "No new media/state reads, decoding, remote checks or physical QA.",
            "v2 per-point groups do not enforce the new continuous-interval rule.",
            "Known same-case cross-engine links are not complete verified configuration matching.",
            "Prior metric development exposure persists after changing event holdouts.",
        ],
    }
    return summary, routing


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, default=repo / "docs/joint_split_v3/plan.json")
    ap.add_argument("--inventory", type=Path, default=repo.parent / "dataset_splits_v2/all_selected_trajectories.jsonl")
    ap.add_argument("--development", type=Path, default=repo / "docs/phyediting/sampling_audit15.json")
    ap.add_argument("--catalog-root", type=Path, default=repo.parent / "hf_audit_20261002")
    ap.add_argument("--out", type=Path, default=repo / "docs/joint_split_v3")
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.inventory.read_text(encoding="utf-8").splitlines() if l.strip()]
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    dev = json.loads(args.development.read_text(encoding="utf-8"))["records"]
    summary, routing = audit(plan, rows, dev)
    recovered, catalog_sources = recover_missing_references(rows, args.catalog_root)
    summary["sidecar_recovery"] = {
        "attempted_records": len(recovered),
        "records_with_catalog_candidates": sum(bool(r["sidecar_candidates"]) for r in recovered),
        "records_with_state_or_metadata_candidates": sum(bool(r["state_or_metadata_candidates"]) for r in recovered),
        "content_validation": "not_read_or_verified",
    }
    summary["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (args.plan, args.inventory, args.development)}
    summary["catalog_source_sha256"] = catalog_sources
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "inventory_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out / "event_pool_routing.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in routing:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.out / "recovered_sidecar_references.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in recovered:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("target_counts", "inventory_total_videos", "inventory_pool_videos", "quarantined_records", "known_metric_development_items_by_new_pool")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
