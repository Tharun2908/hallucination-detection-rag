"""Selection/leakage contracts; fixtures only, no GPU or dataset download."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import prepare_pilot as prep
from post_thesis.llm_judge.judge import JudgeConfig, build_request
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import RunConflict


def fixture():
    return [{"id": f"{group}-{variant}", "context": f"Source {group}: full evidence.",
             "output": f"Answer {variant}", "task_type": "QA", "model": "offline-generator",
             "quality": "good", "query": "QUESTION MUST NOT ENTER REQUEST",
             "hallucination_labels_processed": {"evident_conflict": variant, "baseless_info": 0}}
            for group in range(8) for variant in range(2)]


def bundle(rows=None):
    return prep.build_bundle(fixture() if rows is None else rows, revision="fixture-commit", pilot_size=3)


def reseal(value):
    value["manifest_sha256"] = content_hash(value["manifest"])
    return value


class PilotTests(unittest.TestCase):
    def test_group_reservation_blocks_every_sibling(self):
        value = bundle()
        self.assertEqual(len(prep.pilot_examples(value)), 3)
        rows = value["manifest"]["train_reservation"]
        pilot_groups = {r["group_sha256"] for r in rows if r["partition"] == "pilot"}
        self.assertEqual(sum(r["partition"] == "pilot_group_excluded" for r in rows), 3)
        for row in rows:
            self.assertEqual(row["group_sha256"] not in pilot_groups,
                             row["partition"] == "threshold_candidate")

    def test_selection_is_label_and_metadata_blind_and_order_independent(self):
        rows = fixture()
        original = bundle(rows)["manifest"]["pilot_inputs"]
        for row in rows:
            row["hallucination_labels_processed"] = {"evident_conflict": 0, "baseless_info": 20}
            row.update(task_type="Summary", model="different", quality="truncated")
        self.assertEqual(original, bundle(rows)["manifest"]["pilot_inputs"])
        self.assertEqual(original, bundle(list(reversed(rows)))["manifest"]["pilot_inputs"])

    def test_normalization_changes_group_only_and_preserves_actual_evidence(self):
        rows = fixture()
        rows[1]["context"] = "Source\u00a0０:   full evidence.\n"
        value = bundle(rows)
        reservation = value["manifest"]["train_reservation"]
        self.assertEqual(reservation[0]["group_sha256"], reservation[1]["group_sha256"])
        original = {r["id"]: r for r in rows}
        for item in prep.pilot_examples(value):
            self.assertEqual(item.item.context, original[item.sample_id]["context"])

    def test_real_request_contains_only_answer_and_context(self):
        config = JudgeConfig(provider="offline", model="fixture", api_config_id="fixture")
        for example in prep.pilot_examples(bundle()):
            request = build_request(example.item, config=config)
            payload = json.loads(request.messages[1].content)
            self.assertEqual(payload, asdict(example.item))
            self.assertEqual(set(payload), {"answer", "context"})

    def test_invalid_rows_fail_instead_of_silent_drop(self):
        mutations = [lambda r: r.update(id=""), lambda r: r.update(output=" "),
                     lambda r: r.update(context=""), lambda r: r.update(context=None),
                     lambda r: r.update(hallucination_labels_processed={"evident_conflict": True, "baseless_info": 0}),
                     lambda r: r.update(hallucination_labels_processed={"evident_conflict": -1, "baseless_info": 0})]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                rows = fixture()
                mutate(rows[0])
                with self.assertRaises((ValueError, TypeError)):
                    bundle(rows)
        rows = fixture()
        rows[1]["id"] = rows[0]["id"]
        with self.assertRaises(ValueError):
            bundle(rows)

    def test_target_mapping_includes_either_error_type(self):
        rows = fixture()
        rows[0]["hallucination_labels_processed"] = {"evident_conflict": 0, "baseless_info": 2}
        records, _ = prep._records(rows)
        self.assertEqual(records[0]["label"], 1)
        self.assertEqual(records[1]["label"], 1)
        self.assertEqual(records[2]["label"], 0)

    def test_tamper_and_extra_prompt_fields_are_rejected(self):
        value = bundle()
        value["manifest"]["pilot_inputs"][0]["input"]["answer"] += " changed"
        with self.assertRaises(RunConflict):
            prep.pilot_examples(value)
        with self.assertRaises(RunConflict):
            prep.pilot_examples(reseal(value))
        value = bundle()
        value["manifest"]["pilot_inputs"][0]["input"]["label"] = 1
        with self.assertRaises(RunConflict):
            prep.pilot_examples(reseal(value))

    def test_overlap_is_rejected_even_with_valid_outer_checksum(self):
        value = bundle()
        rows = value["manifest"]["train_reservation"]
        sibling = next(r for r in rows if r["partition"] == "pilot_group_excluded")
        sibling["partition"] = "threshold_candidate"
        with self.assertRaises(RunConflict):
            prep.pilot_examples(reseal(value))

    def test_repeated_preparation_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = bundle()
            self.assertTrue(prep.save_bundle(directory, value))
            before = (directory / "manifest.json").read_bytes()
            self.assertFalse(prep.save_bundle(directory, value))
            changed = deepcopy(value)
            changed["manifest"]["code_revision"] = "changed"
            with self.assertRaises(RunConflict):
                prep.save_bundle(directory, reseal(changed))
            self.assertEqual(before, (directory / "manifest.json").read_bytes())

    def test_wrong_dataset_file_rejected_before_arrow_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.parquet"
            path.write_bytes(b"wrong split")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prep.read_train(path)

    def test_insufficient_groups_are_not_replaced_with_duplicate_samples(self):
        with self.assertRaises(ValueError):
            prep.build_bundle(fixture(), revision="fixture", pilot_size=8)


class ArrowIntegrationTests(unittest.TestCase):
    def test_parquet_reader_and_row_count_gate(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("optional pyarrow integration runs in CPU data CI")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.parquet"
            rows = fixture()
            pq.write_table(pa.Table.from_pylist(rows), path)
            with patch.object(prep, "TRAIN_SHA256", prep.file_sha256(path)):
                with self.assertRaisesRegex(ValueError, "TRAIN rows"):
                    prep.read_train(path)
                with patch.object(prep, "TRAIN_ROWS", len(rows)):
                    loaded = prep.read_train(path)
            self.assertEqual(bundle(rows), bundle(loaded))


if __name__ == "__main__":
    unittest.main()
