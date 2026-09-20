"""Leakage, identity, input isolation and replay checks for TRAIN reservation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import prepare_development as d
from post_thesis.llm_judge.prepare_pilot import build_bundle, group_hash
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import RunConflict


def fixture():
    design = deepcopy(d.load_design())
    design.update(expected_train_rows=8, expected_excluded_rows=2, expected_candidate_rows=6)
    design["allocation"].update(calibration_components=1, operating_threshold_components=1)
    mapping = [{"sample_id": str(i), "source_id": str(i // 2),
                "component_sha256": content_hash([str(i // 2)]),
                "original_partition": ("pilot" if i == 0 else "pilot_group_excluded" if i == 1
                                       else "threshold_candidate"),
                "proposed_excluded": i < 2, "linked_to_pilot": i < 2,
                "linked_to_native_test": False} for i in range(8)]
    return design, mapping


class AllocationTests(unittest.TestCase):
    def test_exact_registered_hash_ranking_and_all_siblings(self):
        design, mapping = fixture()
        reservations, ranking = d.allocate(mapping, design)
        expected = sorted({r["component_sha256"] for r in mapping if not r["proposed_excluded"]},
                          key=lambda c: (hashlib.sha256(
                              ("ragtruth-development-reservation-v1\n20260920\n" + c).encode()).hexdigest(), c))
        self.assertEqual([r["component_sha256"] for r in ranking], expected)
        for name, component in zip(("calibration", "operating_threshold"), expected):
            members = [r for r in reservations if r["partition"] == name]
            self.assertEqual(len(members), 2)
            self.assertEqual({r["component_sha256"] for r in members}, {component})
        self.assertEqual([r["sample_id"] for r in reservations], list(map(str, range(8))))
        self.assertEqual(d.allocate(list(reversed(mapping)), design), (reservations, ranking))

    def test_metadata_and_label_changes_cannot_change_assignment(self):
        design, mapping = fixture()
        before, ranks = d.allocate(mapping, design)
        changed = deepcopy(mapping)
        for row in changed:
            row.update(label=1, task="changed", model="changed", score=999)
        after, after_ranks = d.allocate(changed, design)
        self.assertEqual(ranks, after_ranks)
        self.assertEqual([(r["sample_id"], r["partition"]) for r in before],
                         [(r["sample_id"], r["partition"]) for r in after])

    def test_test_link_cannot_remain_eligible(self):
        design, mapping = fixture()
        mapping[2]["linked_to_native_test"] = True
        with self.assertRaisesRegex(RunConflict, "eligible row"):
            d.allocate(mapping, design)

    def test_original_exclusion_cannot_become_eligible(self):
        design, mapping = fixture()
        mapping[0].update(proposed_excluded=False, linked_to_pilot=False)
        with self.assertRaisesRegex(RunConflict, "eligible row"):
            d.allocate(mapping, design)

    def test_duplicate_invalid_component_and_split_source_rejected(self):
        design, mapping = fixture()
        for field, value in (("sample_id", "0"), ("component_sha256", "BAD"),
                             ("source_id", mapping[0]["source_id"]), ("linked_to_pilot", 0)):
            bad = deepcopy(mapping)
            bad[2][field] = value
            with self.subTest(field=field), self.assertRaises(RunConflict):
                d.allocate(bad, design)

    def test_inconsistent_components_and_counts_rejected(self):
        design, mapping = fixture()
        mapping[2].update(proposed_excluded=True, linked_to_native_test=True)
        with self.assertRaisesRegex(RunConflict, "inconsistent"):
            d.allocate(mapping, design)
        design, mapping = fixture()
        with self.assertRaisesRegex(RunConflict, "counts"):
            d.allocate(mapping[:-1], design)
        design["allocation"]["calibration_components"] = 10
        with self.assertRaisesRegex(RunConflict, "insufficient"):
            d.allocate(mapping, design)

    def test_complete_component_can_contain_multiple_native_sources(self):
        design, mapping = fixture()
        mapping[3]["source_id"] = "extra-native-source"
        rows, _ = d.allocate(mapping, design)
        self.assertEqual(rows[2]["partition"], rows[3]["partition"])

    def test_design_and_report_hash_drift_rejected(self):
        design, _ = fixture()
        with patch.object(d, "DESIGN_SHA256", "changed"), self.assertRaises(RunConflict):
            d.load_design()
        report = {"code_revision": design["cross_split_code_revision"],
                  "train_report_sha256": design["train_report_sha256"]}
        design["cross_split_report_sha256"] = content_hash(report)
        bundle = {"report_sha256": content_hash(report), "report": report}
        prior = {"report_sha256": design["train_report_sha256"]}
        with patch.object(d, "validate_train_report", return_value={}):
            d.validate_reports(bundle, prior, design)
            bundle["report"]["code_revision"] = "drift"
            with self.assertRaisesRegex(RunConflict, "cross-split"):
                d.validate_reports(bundle, prior, design)

    def test_atomic_identical_replay_and_changed_manifest_refusal(self):
        manifest = {"study_stage": "post_thesis", "sample_ids": ["1", "2"]}
        bundle = {"manifest": manifest, "manifest_sha256": content_hash(manifest)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertTrue(d.save_manifest(path, bundle))
            before = (path / "manifest.json").read_bytes()
            with patch.object(d, "atomic_json", side_effect=AssertionError("replay rewrote file")):
                self.assertFalse(d.save_manifest(path, bundle))
            changed = deepcopy(bundle)
            changed["manifest"]["sample_ids"].append("3")
            with self.assertRaisesRegex(RunConflict, "checksum"):
                d.save_manifest(path, changed)
            changed["manifest_sha256"] = content_hash(changed["manifest"])
            with self.assertRaisesRegex(RunConflict, "preserve"):
                d.save_manifest(path, changed)
            self.assertEqual(before, (path / "manifest.json").read_bytes())

    def test_manifest_input_isolation_missing_class_and_historical_integrity(self):
        # Exercise the manifest builder with small genuine pilot structures;
        # patch only external frozen identities and the pilot's sample size.
        rows = [{"id": str(i), "output": f"Answer {i}", "context": f"Context {i // 2}",
                 "hallucination_labels_processed": {"evident_conflict": 0, "baseless_info": 0},
                 "task_type": "QA", "model": "generator-private", "quality": "good"}
                for i in range(8)]
        pilot = build_bundle(rows, revision=d.PREPARATION_REVISION, pilot_size=1)
        original = {r["sample_id"]: r["partition"] for r in pilot["manifest"]["train_reservation"]}
        design, mapping = fixture()
        for row in mapping:
            row["original_partition"] = original[row["sample_id"]]
            excluded = row["original_partition"] != "threshold_candidate"
            row.update(proposed_excluded=excluded, linked_to_pilot=excluded)
        design["pilot_manifest_sha256"] = pilot["manifest_sha256"]
        exclusions = [r["sample_id"] for r in mapping if r["proposed_excluded"]]
        prior = {"proposed_exclusion_ids": exclusions, "train_mapping": [
            {**r, "context_group_sha256": group_hash(rows[int(r["sample_id"])]["context"])}
            for r in mapping]}
        cross = {"train_mapping": mapping, "proposed_exclusion_ids": exclusions,
                 "missing_business_source_ids": ["1"], "native_revision": "pinned",
                 "native_files": {}, "not_covered": ["fuzzy overlap"]}
        before = deepcopy((rows, pilot, cross, prior))
        def pilot_builder(data, revision):
            return build_bundle(data, revision=revision, pilot_size=1)
        with patch.object(d, "load_design", return_value=design), \
                patch.object(d, "validate_reports", return_value=(cross, prior)), \
                patch.object(d, "build_bundle", side_effect=pilot_builder):
            result = d.build_manifest(rows, pilot, {}, {}, revision="local-test")
            manifest = result["manifest"]
            self.assertEqual(manifest["status"], "reserved_missing_class_do_not_fit")
            self.assertFalse(manifest["scoring_authorized"])
            self.assertEqual(set(manifest["model_inputs"]), {"calibration", "operating_threshold"})
            for arm in manifest["model_inputs"].values():
                for record in arm:
                    self.assertEqual(set(record), {"sample_id", "answer", "context"})
                    self.assertNotIn("generator-private", json.dumps(record))
            self.assertTrue(manifest["checks"]["component_disjoint"])
            self.assertEqual(result["manifest_sha256"], content_hash(manifest))
            bad = deepcopy(rows)
            bad[0]["output"] += " drift"
            with self.assertRaisesRegex(RunConflict, "original pilot"):
                d.build_manifest(bad, pilot, {}, {}, revision="local-test")
        self.assertEqual((rows, pilot, cross, prior), before)


if __name__ == "__main__":
    unittest.main()
