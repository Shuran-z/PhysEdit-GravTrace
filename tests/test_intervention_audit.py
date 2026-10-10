"""Guard against inflating intervention counts by merging incompatible samples."""
from copy import deepcopy
import unittest

from scripts.phyediting_intervention_audit import audit, parse_label


def triple():
    return [dict(record_id=b, sample_id=f"T01V2__bg01__g01__cfA_theta_{b}_rep01",
                 engine="genesis", source_kind="huggingface", repo_id="repo", revision="rev",
                 release="release", event_id="T01", background_id="bg01", gravity_id="g01",
                 source_group_key="T01|g01|cfA", split="train", leakage_group_id="group",
                 sidecar_paths=["metadata.json"], videos={v: {} for v in ("cam01", "cam02", "cam03")})
            for b in ("minus", "star", "plus")]


class InterventionAuditTests(unittest.TestCase):
    def test_complete_does_not_certify_physics_or_unseen(self):
        s, groups = audit(triple(), set())
        self.assertEqual(s["complete_named_triples"], 1)
        self.assertTrue(groups[0]["structural_review_candidate"])
        self.assertEqual(s["physics_certification"], "not_performed")
        self.assertEqual(groups[0]["unseen_status"], "not_certified_by_this_audit")

    def test_cannot_complete_across_version_background_or_gravity(self):
        for key in ("release", "revision", "background_id", "gravity_id"):
            with self.subTest(key=key):
                rows = triple(); rows[-1][key] = "different"
                self.assertEqual(audit(rows, set())[0]["complete_named_triples"], 0)

    def test_duplicates_are_ambiguous_not_extra_triples(self):
        rows = triple(); duplicate = deepcopy(rows[-1]); duplicate["record_id"] = "other"
        s, groups = audit(rows + [duplicate], set())
        self.assertEqual(s["complete_named_triples"], 0)
        self.assertIn("ambiguous_duplicate_branch", groups[0]["review_blockers"])

    def test_cross_split_and_missing_reference_block_review(self):
        rows = triple(); rows[0]["split"] = "test_id"; rows[1]["sidecar_paths"] = []
        s, groups = audit(rows, {"group"})
        self.assertEqual(s["complete_named_triples"], 1)
        self.assertFalse(groups[0]["structural_review_candidate"])
        self.assertTrue(groups[0]["known_metric_development_group"])

    def test_repeats_and_backgrounds_do_not_inflate_family_count(self):
        rows = triple(); extra = deepcopy(rows)
        for r in extra:
            r["record_id"] += "2"; r["background_id"] = "bg02"
        s, _ = audit(rows + extra, set())
        self.assertEqual(s["complete_named_triples"], 2)
        self.assertEqual(s["by_engine_event"]["genesis/T01"]["complete_versioned_families"], 1)

    def test_does_not_guess_ambiguous_repeat(self):
        row = triple()[0]; row["sample_id"] = "T11__g01__A_speed_theta_minus_03"
        self.assertIsNone(parse_label(row)["repeat"])
        self.assertEqual(audit([row], set())[0]["complete_named_triples"], 0)

    def test_server_branch_and_zero_repeat(self):
        row = triple()[0]; row.update(source_kind="ssh", source_slot="boundary:T16__bg01__g01__d1__plus__r0")
        label = parse_label(row)
        self.assertEqual((label["branch"], label["repeat"]), ("plus", 0))


if __name__ == "__main__":
    unittest.main()
