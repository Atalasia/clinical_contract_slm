from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from cpg2 import slm_e3_contract as paired
from cpg2.mimic.csvio import config_sha256


SPEC = importlib.util.spec_from_file_location(
    "report_clinical_note_label_identity", Path(__file__).resolve().parents[1] / "scripts/report_clinical_note_label_identity.py"
)
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)
MODEL = "mg4_it_v1"
YES = {"detection": "yes", "encounter": "yes", "negation": "no"}
OTHER = {"detection": "yes", "encounter": "no", "negation": "yes"}


def row(index, labels=YES, *, raw=None, condition="unconstrained"):
    request = {"request_id": f"private-request-{index}", "expected": dict(YES),
               "note_cluster": f"private-note-{index}", "admission_cluster": f"private-admission-{index // 2}",
               "subject_cluster": f"private-subject-{index // 2}",
               "semantic_group": "invented", "context_mode": "complete_note"}
    return {**request, "manifest_id": MODEL, "condition": condition, "prompt_sha256": str(index) * 64,
            "prompt_token_ids_sha256": config_sha256([index, 2, 3]), "prompt_tokens_preflight": 3,
            "prompt_tokens": 3, "output_tokens": 5, "finish_reason": "stop", "runtime_failed": False,
            "raw_output": json.dumps(labels) if raw is None else raw,
            **paired.evaluate_output(json.dumps(labels) if raw is None else raw, YES)}


class LabelIdentitySemanticsTests(unittest.TestCase):
    def test_same_reference_accuracy_can_have_all_or_no_identical_triples(self):
        left = [row(i, YES if i < 2 else OTHER) for i in range(4)]
        equal = [row(i, YES if i < 2 else OTHER, condition="constrained") for i in range(4)]
        different = [row(i, OTHER if i < 2 else YES, condition="constrained") for i in range(4)]
        self.assertEqual(sum(r["hierarchical_joint_correct"] for r in left), 2)
        self.assertEqual(sum(r["hierarchical_joint_correct"] for r in equal), 2)
        self.assertEqual(sum(r["hierarchical_joint_correct"] for r in different), 2)
        self.assertEqual(identity.summarize_rows(left, equal)["triple_identity"]["identical"], 4)
        self.assertEqual(identity.summarize_rows(left, different)["triple_identity"]["identical"], 0)

    def test_pairing_is_by_id_not_position(self):
        left = [row(i, YES if i < 2 else OTHER) for i in range(4)]
        right = [row(i, YES if i < 2 else OTHER, condition="constrained") for i in reversed(range(4))]
        self.assertEqual(identity.summarize_rows(left, right)["triple_identity"]["identical"], 4)

    def test_invalid_pairs_are_nonmatches_not_dropped_or_equal_null_labels(self):
        left = [row(0), row(1, raw="not JSON"), row(2), row(3, raw="not JSON")]
        right = [row(0), row(1), row(2, raw="not JSON"), row(3, raw="not JSON")]
        result = identity.summarize_rows(left, right)
        self.assertEqual(result["triple_identity"], {"identical": 1, "n": 4, "proportion": 0.25})
        self.assertEqual(result["both_parseable"], 1)
        self.assertEqual(result["pair_status_counts"], {
            "both_valid": 1, "unconstrained_invalid_only": 1, "constrained_invalid_only": 1, "both_invalid": 1})
        self.assertEqual(result["identity_among_both_valid"]["proportion"], 1)
        self.assertEqual(result["conditions"]["unconstrained"]["failure_counts"]["parse_failure"], 2)

    def test_abstention_and_not_applicable_are_literal_labels(self):
        labels = [dict(YES), {"detection": "no", "encounter": "not_applicable", "negation": "not_applicable"},
                  {"detection": "abstain", "encounter": "abstain", "negation": "abstain"},
                  {**YES, "negation": "unsure"}]
        left = [row(i, value) for i, value in enumerate(labels)]
        right = [row(i, value) for i, value in enumerate(labels)]
        result = identity.summarize_rows(left, right)
        self.assertEqual(result["triple_identity"]["identical"], 4)
        self.assertEqual(result["identical_triples_with_abstention"], 1)
        self.assertEqual(result["conditions"]["unconstrained"]["abstention_triples"], 1)
        right[3] = row(3, YES)
        self.assertEqual(identity.summarize_rows(left, right)["triple_identity"]["identical"], 3)

    def test_conservative_marker_handling_is_only_on_unconstrained_arm(self):
        valid = json.dumps(YES)
        fence = "```json\n" + valid + "\n```"
        left = [row(0, raw=fence), row(1, raw="Answer: " + fence), row(2), row(3, raw=fence + "\nExplanation")]
        right = [row(0), row(1), row(2, raw=fence), row(3)]
        result = identity.summarize_rows(left, right)
        self.assertEqual(result["triple_identity"]["identical"], 1)
        self.assertEqual(result["conditions"]["unconstrained"]["fences_removed"], 1)
        self.assertEqual(result["conditions"]["constrained"]["fences_removed"], 0)

    def test_logical_failure_can_have_identical_labels_but_schema_failure_cannot(self):
        raws = [json.dumps({**YES, "extra": "private-sentinel"}),
                json.dumps({**YES, "detection": "no"})]
        left = [row(i, raw=value) for i, value in enumerate(raws)]
        result = identity.summarize_rows(left, copy.deepcopy(left))
        self.assertEqual(result["triple_identity"], {"identical": 1, "n": 2, "proportion": 0.5})
        self.assertEqual(result["both_parseable"], 2)
        self.assertEqual(result["both_schema_valid_label_triples"], 1)
        self.assertEqual(result["both_valid_label_triples"], 0)
        self.assertEqual(result["identity_among_both_valid"]["identical"], 0)
        self.assertIsNone(result["identity_among_both_valid"]["proportion"])
        self.assertEqual(result["conditions"]["unconstrained"]["failure_counts"]["schema_failure"], 1)
        self.assertEqual(result["conditions"]["unconstrained"]["failure_counts"]["logical_failure"], 1)

    def test_label_identity_and_logical_identity_use_separate_numerators(self):
        raws = [json.dumps(YES), json.dumps({**YES, "detection": "no"}),
                json.dumps({**YES, "extra": "private-sentinel"}), "not JSON"]
        left = [row(i, raw=value) for i, value in enumerate(raws)]
        result = identity.summarize_rows(left, copy.deepcopy(left))
        self.assertEqual(result["triple_identity"], {"identical": 2, "n": 4, "proportion": 0.5})
        self.assertEqual(result["both_schema_valid_label_triples"], 2)
        self.assertEqual(result["identity_among_both_valid"], {"identical": 1, "n": 1, "proportion": 1.0})
        self.assertEqual(result["pair_status_counts"]["both_invalid"], 3)
        for head in identity.HEADS:
            self.assertEqual(result["per_head_identity"][head], {"identical": 2, "n": 4, "proportion": 0.5})

    def test_missing_or_duplicate_id_rejected(self):
        for right in ([row(0)], [row(0), row(0)]):
            with self.subTest(), self.assertRaises(ValueError):
                identity.summarize_rows([row(0), row(1)], right)


class LabelIdentityArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_dir = self.root / "configs"
        self.config_dir.mkdir()
        self.run = self.root / "outputs/local_restricted/run"
        self.dataset_dir = self.root / "outputs/local_restricted/dataset"
        self.dataset_dir.mkdir(parents=True)
        self.config_path = self.config_dir / "paired.json"
        self.cells = {condition: [row(i, condition=condition) for i in range(4)] for condition in paired.CONDITIONS}
        self.requests = [{key: value for key, value in r.items() if key in ("request_id", *paired.METADATA_FIELDS)}
                         for r in self.cells["unconstrained"]]
        for request in self.requests:
            request["note_context"] = "private-clinical-text-sentinel"
        self.counts = {"requests": 4, "notes": 4, "admissions": 2, "subjects": 2}
        self.wrapper = {"mode": "official_chat_template", "messages": []}
        self.write(self.config_dir / "wrapper.json", self.wrapper)
        self.model = {"manifest_id": MODEL, "model": {"model_id": "invented-model", "revision": "invented-revision",
                       "wrapper_spec": "wrapper.json"}, "verification": {"artifact_sha256": "a" * 64,
                       "wrapper_spec_sha256": identity.digest(self.config_dir / "wrapper.json")}}
        self.write(self.config_dir / "model.json", self.model)
        self.benchmark = {"model_manifests": ["model.json"]}
        self.write(self.config_dir / "benchmark.json", self.benchmark)
        self.prompt = {"output_schema": {"type": "object"}, "user_template": "invented"}
        self.write(self.config_dir / "prompt.json", self.prompt)
        self.primary = {"freeze_status": "FROZEN", "experiment_id": "invented_primary", "panel": [MODEL],
                        "benchmark_config": "benchmark.json", "benchmark_config_sha256": config_sha256(self.benchmark),
                        "prompt": "prompt.json", "prompt_sha256": config_sha256(self.prompt),
                        "generation": {"temperature": 0, "seed": 2026, "max_output_tokens": 96},
                        "context_policy": {"max_context_tokens": 2048},
                        "dataset": {"name": "invented-dataset", "version": "1.0.0", "expected_counts": self.counts,
                                    "notes_sha256": "b" * 64, "labels_sha256": "c" * 64, "source_audit_sha256": "d" * 64}}
        self.write(self.config_dir / "primary.json", self.primary)
        self.requests_path = self.dataset_dir / "requests.jsonl"
        self.write(self.requests_path, self.requests, jsonl=True)
        self.manifest = {"adapter_stage": "slm_benchmark_e3_prepare", "experiment_id": "invented_primary",
                         "e3_config_sha256": config_sha256(self.primary), "dataset": "invented-dataset", "dataset_version": "1.0.0",
                         "source_files": {key: self.primary["dataset"][key] for key in ("notes_sha256", "labels_sha256", "source_audit_sha256")},
                         "benchmark_config_sha256": config_sha256(self.benchmark), "prompt_sha256": config_sha256(self.prompt),
                         "counts": self.counts, "output_requests": str(self.requests_path.relative_to(self.root)),
                         "output_requests_sha256": identity.digest(self.requests_path)}
        self.manifest_path = self.dataset_dir / "manifest.json"
        self.write(self.manifest_path, self.manifest)
        self.config = {"schema_version": "1.0", "experiment_id": "invented_paired", "freeze_status": "FROZEN",
                       "primary_e3_config": "primary.json", "primary_e3_config_sha256": config_sha256(self.primary),
                       "dataset_manifest": "../outputs/local_restricted/dataset/manifest.json",
                       "dataset_manifest_sha256": identity.digest(self.manifest_path), "conditions": list(paired.CONDITIONS),
                       "expected_counts": self.counts, "checkpoint_batch_size": 2, "wrapper_overrides": {},
                       "structured_backend": "xgrammar", "runtime_versions": {"vllm": "archived-not-installed"},
                       "engine_overrides": {"enable_prefix_caching": False, "add_special_tokens": False},
                       "output_root": "../outputs/local_restricted/run"}
        self.write(self.config_path, self.config)
        self.provenance = {
            "experiment_id": self.config["experiment_id"], "experiment_config_sha256": config_sha256(self.config),
            "dataset_manifest_sha256": identity.digest(self.manifest_path), "requests_sha256": identity.digest(self.requests_path),
            "manifest_id": MODEL, "manifest_sha256": config_sha256(self.model), "model_revision": "invented-revision",
            "previously_verified_artifact_sha256": "a" * 64, "wrapper_sha256": identity.digest(self.config_dir / "wrapper.json"),
            "prompt_sha256": config_sha256(self.prompt), "schema_sha256": config_sha256(self.prompt["output_schema"]),
            "runtime_versions": self.config["runtime_versions"], "inference_source_sha256": {name: "e" * 64 for name in identity.SOURCE_NAMES},
            "structured_backend": "xgrammar", "generation": self.primary["generation"],
            "decoder_override": "constrained_decoding enabled only in constrained cell", "checkpoint_batch_size": 2,
            "engine_overrides": self.config["engine_overrides"],
        }
        self.preflight = {"experiment_config_sha256": config_sha256(self.config), "all_models_fit_context": True, "models": {MODEL: {
            "provenance": self.provenance, "counts": self.counts, "over_context": 0, "complete_object_no_prefill": True,
            "output_token_budget": 96, "prompt_tokens_min": 3, "prompt_tokens_max": 3, "prompt_tokens_mean": 3,
            "rendered_prompt_fingerprint": config_sha256([[r["request_id"], r["prompt_sha256"], r["prompt_token_ids_sha256"],
                                                           r["prompt_tokens_preflight"]] for r in self.cells["unconstrained"]])}}}
        self.write(self.run / "preflight.json", self.preflight)
        self.save_cells()

    @staticmethod
    def write(path, value, *, jsonl=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(("\n".join(json.dumps(row) for row in value) if jsonl else json.dumps(value)) + "\n", encoding="utf-8")
        path.chmod(0o600)

    def save_cells(self):
        for condition, rows in self.cells.items():
            folder = self.run / "results" / MODEL / condition
            self.write(folder / "responses.jsonl", rows, jsonl=True)
            batches = []
            for index in range(2):
                batch_rows = [r for r in rows if r["request_id"] in {f"private-request-{index * 2}", f"private-request-{index * 2 + 1}"}]
                value = {"provenance": self.provenance, "condition": condition, "batch_index": index,
                         "technical_completion": True, "rows": batch_rows, "rows_sha256": config_sha256(batch_rows)}
                path = folder / "batches" / f"batch_{index:05d}.json"
                self.write(path, value)
                batches.append(identity.digest(path))
            self.write(folder / "audit.json", {"provenance": self.provenance, "condition": condition,
                       "technical_completion": True, "n": 4, "batches": 2,
                       "responses_sha256": identity.digest(folder / "responses.jsonl"), "batch_sha256": batches})

    def report(self, **kwargs):
        return identity.build_report(self.config_path, repository_root=self.root, **kwargs)

    def test_complete_archived_run_needs_no_current_inference_runtime_or_source_equality(self):
        with patch.object(paired, "_versions", side_effect=AssertionError("must not query inference runtime")), \
             patch.object(paired, "_load_engine", side_effect=AssertionError("must not load model")), \
             patch.object(paired, "_render_requests", side_effect=AssertionError("must not load tokenizer")):
            result = self.report()
        self.assertEqual(result["triple_identity"], {"identical": 4, "n": 4, "proportion": 1})
        self.assertEqual(result["provenance"]["saved_generation_provenance"]["inference_source_sha256"],
                         {name: "e" * 64 for name in identity.SOURCE_NAMES})
        self.assertNotEqual(result["provenance"]["current_analysis_source_sha256"]["src/cpg2/slm_e3_contract.py"], "e" * 64)
        for private in ("private-request", "private-note", "private-admission", "private-subject", "private-clinical-text-sentinel"):
            self.assertNotIn(private, json.dumps(result))

    def test_saved_rows_can_be_reordered_with_correct_checksums(self):
        self.cells["constrained"].reverse()
        self.save_cells()
        self.assertEqual(self.report()["triple_identity"]["identical"], 4)

    def test_invalid_saved_outputs_are_retained_in_all_request_denominator(self):
        self.cells["unconstrained"][2] = row(2, raw="private-invalid-output-sentinel")
        self.save_cells()
        result = self.report()
        self.assertEqual(result["triple_identity"], {"identical": 3, "n": 4, "proportion": 0.75})
        self.assertNotIn("private-invalid-output-sentinel", json.dumps(result))

    def test_forged_saved_evaluation_is_rejected_even_after_rehashing(self):
        self.cells["constrained"][0]["parsed"]["negation"] = "yes"
        self.save_cells()
        with self.assertRaisesRegex(ValueError, "stored validation or scores"):
            self.report()

    def test_mismatched_metadata_and_prompt_are_rejected_after_rehashing(self):
        original = copy.deepcopy(self.cells)
        changes = {"subject_cluster": "private-wrong-subject", "expected": OTHER,
                   "prompt_sha256": "f" * 64, "prompt_token_ids_sha256": "f" * 64,
                   "manifest_id": "wrong-model", "condition": "wrong-condition"}
        for field, value in changes.items():
            with self.subTest(field=field):
                self.cells = copy.deepcopy(original)
                self.cells["constrained"][0][field] = value
                self.save_cells()
                with self.assertRaises(ValueError) as caught:
                    self.report()
                self.assertNotIn("private-", str(caught.exception))

    def test_missing_and_duplicate_rows_are_rejected(self):
        original = copy.deepcopy(self.cells)
        for rows in (original["constrained"][:-1], original["constrained"] + [original["constrained"][0]]):
            with self.subTest():
                self.cells = copy.deepcopy(original)
                self.cells["constrained"] = rows
                self.save_cells()
                with self.assertRaises(ValueError):
                    self.report()

    def test_artifact_checksum_mutation_is_rejected(self):
        paths = [self.requests_path, self.run / "results" / MODEL / "constrained/responses.jsonl",
                 self.run / "results" / MODEL / "constrained/batches/batch_00000.json"]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "hash differs|checksum differs"):
                    self.report()
                path.write_bytes(original)

    def test_cross_arm_generation_provenance_mismatch_rejected(self):
        path = self.run / "results" / MODEL / "constrained/audit.json"
        audit = identity.read(path)
        audit["provenance"]["inference_source_sha256"]["slm_e3.py"] = "f" * 64
        self.write(path, audit)
        with self.assertRaisesRegex(ValueError, "audit provenance"):
            self.report()

    def test_bad_preflight_configuration_and_fingerprint_rejected(self):
        original = copy.deepcopy(self.preflight)
        for change in ("config", "fingerprint", "source_hashes", "context_status", "token_lengths"):
            with self.subTest(change=change):
                value = copy.deepcopy(original)
                if change == "config":
                    value["experiment_config_sha256"] = "f" * 64
                elif change == "fingerprint":
                    value["models"][MODEL]["rendered_prompt_fingerprint"] = "f" * 64
                elif change == "source_hashes":
                    value["models"][MODEL]["provenance"]["inference_source_sha256"] = {}
                elif change == "context_status":
                    value["all_models_fit_context"] = False
                else:
                    value["models"][MODEL]["prompt_tokens_max"] = 99
                self.write(self.run / "preflight.json", value)
                with self.assertRaises(ValueError):
                    self.report()

    def test_runtime_failed_cell_cannot_claim_technical_completion(self):
        self.cells["constrained"][0]["runtime_failed"] = True
        self.save_cells()
        with self.assertRaisesRegex(ValueError, "runtime-failed responses"):
            self.report()

    def test_selected_model_and_alternate_run_root(self):
        self.assertEqual(self.report(run_root=self.run)["manifest_id"], MODEL)
        with self.assertRaisesRegex(ValueError, "selected model"):
            self.report(manifest_id="unknown")

    def test_report_is_restricted_idempotent_and_non_overwriting(self):
        result = self.report()
        path = self.run / "diagnostics/result.json"
        identity.write_report(path, result, repository_root=self.root)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        original = path.read_bytes()
        identity.write_report(path, result, repository_root=self.root)
        with self.assertRaisesRegex(ValueError, "existing artifact differs"):
            identity.write_report(path, {"changed": True}, repository_root=self.root)
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "must remain under"):
            identity.write_report(self.root / "public.json", result, repository_root=self.root)


if __name__ == "__main__":
    unittest.main()
