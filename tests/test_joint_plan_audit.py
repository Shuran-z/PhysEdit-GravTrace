from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.phyediting_joint_plan_audit import audit, catalog_sample_ids, target_counts, validate_plan

PLAN = json.loads((Path(__file__).resolve().parents[1] / "docs/joint_split_v3/plan.json").read_text(encoding="utf-8"))


def row(event, engine="genesis", group="g", record="r"):
    return dict(record_id=record, sample_id=event + "_example", event_id=event,
                engine=engine, leakage_group_id=group, split="train",
                videos={k: {"path": f"{record}/{k}.mp4"} for k in ("cam01", "cam02", "cam03")})


class JointPlanTests(unittest.TestCase):
    def test_exact_user_capacity(self):
        counts = target_counts(PLAN)
        self.assertEqual([counts[k]["total"] for k in ("train", "val", "test_id", "test_ood_1", "test_ood_2")],
                         [47040, 5880, 5880, 6240, 7920])
        self.assertEqual(sum(v["total"] for v in counts.values()), 72960)

    def test_duplicate_event_pool_rejected(self):
        p = deepcopy(PLAN); p["train_pool"].append("T03")
        with self.assertRaises(ValueError): validate_plan(p)

    def test_route_overrides_old_split_for_both_engines(self):
        rows = [row("T03", "genesis", record="r1"), row("T03", "isaac", record="r2")]
        s, rr = audit(PLAN, rows, [])
        self.assertEqual({r["planned_pool"] for r in rr}, {"test_ood_1"})
        self.assertEqual(s["quarantined_records"], 0)
        self.assertTrue(all(r["assignment_status"] != "train" for r in rr))

    def test_cross_pool_link_quarantined(self):
        s, rr = audit(PLAN, [row("T01", record="r1"), row("T03", record="r2")], [])
        self.assertEqual(s["quarantined_records"], 2)
        self.assertTrue(all(r["route_status"] == "quarantine_pool_conflict" for r in rr))

    def test_identical_media_cannot_cross_pools(self):
        a, b = row("T01", group="a", record="a"), row("T19", group="b", record="b")
        a["videos"]["cam01"]["sha256"] = b["videos"]["cam01"]["sha256"] = "same"
        self.assertEqual(audit(PLAN, [a, b], [])[0]["quarantined_records"], 2)

    def test_missing_states_and_zero_targets_not_fabricated(self):
        s, _ = audit(PLAN, [row("T01")], [])
        t11 = next(r for r in s["event_inventory"] if r["event_id"] == "T11" and r["engine"] == "isaac")
        self.assertTrue(t11["zero_target_is_intentional"])
        self.assertEqual(t11["nominal_shortfall_videos"], 0)
        self.assertIsNone(t11["actual_usable_videos"])

    def test_old_metric_exposure_is_not_erased(self):
        s, _ = audit(PLAN, [], [{"id": "T03__old"}, {"id": "T19__old"}])
        self.assertEqual(s["known_metric_development_items_by_new_pool"], {"test_ood_1": 1, "test_ood_2": 1})

    def test_sidecar_matching_uses_exact_identity(self):
        self.assertIn("T03__sample", catalog_sample_ids("release/metadata/T03__sample__cam01.json.gz"))
        self.assertIn("T03__sample", catalog_sample_ids("release/trajectories/T03__sample/state.json"))
        self.assertNotIn("T03__sample", catalog_sample_ids("release/metadata/T03__sample_extra__cam01.json.gz"))


if __name__ == "__main__":
    unittest.main()
