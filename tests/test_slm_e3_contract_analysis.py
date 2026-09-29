from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from cpg2.ext_notes_benchmark import parse_ext_notes_output
from cpg2.slm_e3 import _hierarchical_correct, _prediction_heads
from cpg2.slm_e3_contract_analysis import summarize_paired_e3


GOLD = {"detection": "yes", "encounter": "yes", "negation": "no"}
WRONG = {"detection": "no", "encounter": "not_applicable", "negation": "not_applicable"}
ABSTAIN = {"detection": "abstain", "encounter": "abstain", "negation": "abstain"}
CONTRASTS = [{"contrast_id": "a_to_b", "models": ["model_a", "model_b"]}]


def _evaluation(raw, expected, *, runtime_failed=False):
    parsed = parse_ext_notes_output(raw, runtime_failed=runtime_failed).as_dict()
    parsed["parse_valid"] = parsed["failure"] not in {"runtime_failure", "parse_failure"}
    return {"parsed": parsed, "predictions": _prediction_heads(parsed),
            "hierarchical_joint_correct": _hierarchical_correct(expected, parsed)}


def _row(index, prediction, *, subject="subject_private_0", expected=None,
         normalized_prediction=None, runtime_failed=False):
    expected = copy.deepcopy(GOLD if expected is None else expected)
    raw = json.dumps(prediction) if isinstance(prediction, dict) else prediction
    normalized = prediction if normalized_prediction is None else normalized_prediction
    normalized_raw = json.dumps(normalized) if isinstance(normalized, dict) else normalized
    return {
        "request_id": f"request_private_{index}",
        "subject_cluster": subject,
        "admission_cluster": f"admission_private_{subject}",
        "note_cluster": f"note_private_{subject}",
        "prompt_sha256": "a" * 64,
        "expected": expected,
        "raw_output": raw,
        "runtime_failed": runtime_failed,
        "finish_reason": "missing_output" if runtime_failed else "stop",
        **_evaluation(raw, expected, runtime_failed=runtime_failed),
        "fence_normalized": _evaluation(normalized_raw, expected, runtime_failed=runtime_failed),
    }


def _rows(bits, *, subjects=None):
    subjects = subjects or [f"subject_private_{index // 2}" for index in range(len(bits))]
    return [_row(index, GOLD if bit else WRONG, subject=subjects[index])
            for index, bit in enumerate(bits)]


def _panel():
    return {"model_a": {"unconstrained": _rows([1, 1, 1, 0]),
                        "constrained": _rows([1, 0, 0, 0])},
            "model_b": {"unconstrained": _rows([1, 0, 0, 0]),
                        "constrained": _rows([1, 1, 1, 0])}}


class PairedE3AnalysisTests(unittest.TestCase):
    def test_paired_effects_interaction_and_observed_reversal(self):
        result = summarize_paired_e3(_panel(), CONTRASTS, bootstrap_replicates=100)
        self.assertEqual(result["n_requests"], 4)
        self.assertEqual(result["clusters"], {"notes": 2, "admissions": 2, "subjects": 2})
        self.assertEqual(result["strict"]["within_model_decoding_effects"]["model_a"]["difference"], -0.5)
        self.assertEqual(result["strict"]["within_model_decoding_effects"]["model_b"]["difference"], 0.5)
        contrast = result["strict"]["model_contrasts"][0]
        self.assertEqual(contrast["unconstrained"]["difference"], -0.5)
        self.assertEqual(contrast["constrained"]["difference"], 0.5)
        self.assertEqual(contrast["decoding_interaction"]["difference"], 1.0)
        self.assertTrue(contrast["observed_sign_reversal"])
        self.assertFalse(contrast["observed_tie_in_either_condition"])
        self.assertEqual(result["strict"]["winners"], {
            "unconstrained": ["model_a"], "constrained": ["model_b"]})

    def test_identical_paired_models_have_zero_width_contrast_and_interaction(self):
        panel = _panel()
        panel["model_b"] = copy.deepcopy(panel["model_a"])
        summary = summarize_paired_e3(panel, CONTRASTS, bootstrap_replicates=50)
        contrast = summary["strict"]["model_contrasts"][0]
        for key in ("unconstrained", "constrained", "decoding_interaction"):
            self.assertEqual(contrast[key], {"difference": 0.0, "subject_cluster_ci95": [0.0, 0.0]})
        self.assertFalse(contrast["observed_sign_reversal"])
        self.assertTrue(contrast["observed_tie_in_either_condition"])
        for condition in ("unconstrained", "constrained"):
            self.assertEqual(summary["strict"]["winners"][condition], ["model_a", "model_b"])
            self.assertEqual([row["rank"] for row in summary["strict"]["rankings"][condition]], [1, 1])

    def test_bootstrap_repeats_subjects_with_all_requests_and_common_draws(self):
        subjects = ["s0", "s1", "s1", "s2", "s2", "s2", "s2"]
        bits = {"model_a": {"unconstrained": [1, 1, 0, 0, 0, 0, 0],
                            "constrained": [1, 1, 1, 1, 0, 0, 0]},
                "model_b": {"unconstrained": [0, 1, 1, 1, 1, 0, 0],
                            "constrained": [0, 1, 1, 1, 1, 1, 1]}}
        panel = {model: {condition: _rows(values, subjects=subjects)
                         for condition, values in cells.items()} for model, cells in bits.items()}
        summary = summarize_paired_e3(panel, CONTRASTS, bootstrap_replicates=113, bootstrap_seed=19)
        cells = summary["strict"]["cells"]
        self.assertAlmostEqual(cells["model_a"]["unconstrained"]["hierarchical_joint"]["exact_match"], 2 / 7)
        # Independently materialize each bootstrap sample, including repeated
        # requests. This catches accidental equal-weight subject accuracy and
        # unpaired or condition-specific resampling.
        draws = np.random.default_rng(19).multinomial(3, np.full(3, 1 / 3), size=113)
        samples = {(model, condition): [] for model in bits for condition in bits[model]}
        for counts in draws:
            indexes = [i for s, repetitions in enumerate(counts) for _ in range(repetitions)
                       for i, subject in enumerate(subjects) if subject == f"s{s}"]
            for (model, condition), values in samples.items():
                values.append(sum(bits[model][condition][i] for i in indexes) / len(indexes))
        samples = {key: np.array(values) for key, values in samples.items()}
        for (model, condition), values in samples.items():
            self.assertEqual(cells[model][condition]["hierarchical_joint"]["subject_cluster_ci95"],
                             np.quantile(values, [0.025, 0.975]).tolist())
        u = samples["model_b", "unconstrained"] - samples["model_a", "unconstrained"]
        c = samples["model_b", "constrained"] - samples["model_a", "constrained"]
        contrast = summary["strict"]["model_contrasts"][0]
        self.assertEqual(contrast["decoding_interaction"]["subject_cluster_ci95"],
                         np.quantile(c - u, [0.025, 0.975]).tolist())
        self.assertEqual(summary["strict"]["model_contrasts"], summary["fence_normalized"]["model_contrasts"])

    def test_order_independent_deterministic_and_input_not_mutated(self):
        panel = _panel()
        original = copy.deepcopy(panel)
        first = summarize_paired_e3(panel, CONTRASTS, bootstrap_replicates=75, bootstrap_seed=8)
        second = summarize_paired_e3({model: {condition: list(reversed(rows))
                                            for condition, rows in cells.items()}
                                     for model, cells in reversed(list(panel.items()))},
                                    CONTRASTS, bootstrap_replicates=75, bootstrap_seed=8)
        self.assertEqual(first, second)
        self.assertEqual(panel, original)

    def test_fence_secondary_is_separate_and_counts_failures_without_coercion(self):
        fenced = "```json\n" + json.dumps(GOLD) + "\n```"
        u = [_row(0, fenced, normalized_prediction=GOLD), _row(1, ABSTAIN),
             _row(2, {"detection": "no", "encounter": "yes", "negation": "no"}),
             _row(3, None, runtime_failed=True)]
        panel = {"model_a": {"unconstrained": u, "constrained": copy.deepcopy(u)}}
        result = summarize_paired_e3(panel, [], bootstrap_replicates=5)
        strict = result["strict"]["cells"]["model_a"]["unconstrained"]
        normalized = result["fence_normalized"]["cells"]["model_a"]["unconstrained"]
        self.assertEqual(strict["hierarchical_joint"]["correct"], 0)
        self.assertEqual(normalized["hierarchical_joint"]["correct"], 1)
        self.assertEqual(strict["schema_valid"], 2)
        self.assertEqual(strict["logical_valid"], 1)
        self.assertEqual(strict["abstained"], 1)
        self.assertEqual(strict["runtime_completed"], 3)
        self.assertEqual(strict["failure_counts"], {
            "parse_failure": 1, "logical_failure": 1, "none": 1, "runtime_failure": 1})
        self.assertEqual(strict["detection"]["correct"], 0)

    def test_native_unsure_and_negative_detection_scoring(self):
        unsure = {"detection": "yes", "encounter": "no", "negation": "unsure"}
        rows = [_row(0, unsure, expected=unsure), _row(1, WRONG, expected=WRONG)]
        result = summarize_paired_e3({"model_a": {"unconstrained": rows, "constrained": rows}}, [],
                                    bootstrap_replicates=3)
        metrics = result["strict"]["cells"]["model_a"]["unconstrained"]
        self.assertEqual(metrics["hierarchical_joint"]["exact_match"], 1)
        self.assertEqual(metrics["abstained"], 0)
        self.assertEqual(metrics["negation_gold_detection_yes"]["n"], 1)
        self.assertEqual(metrics["negation_gold_detection_yes"]["per_class"]["unsure"]["true_positive"], 1)

    def test_aggregate_has_no_request_cluster_ids_raw_output_or_prompt_hashes(self):
        summary = summarize_paired_e3(_panel(), CONTRASTS, bootstrap_replicates=5)
        serialized = json.dumps(summary, allow_nan=False)
        for marker in ("request_private", "subject_private", "note_private", "admission_private",
                       "raw_output", "prompt_sha256", "a" * 64):
            self.assertNotIn(marker, serialized)

    def test_different_wrapping_between_models_is_allowed(self):
        panel = _panel()
        for rows in panel["model_b"].values():
            for row in rows:
                row["prompt_sha256"] = "b" * 64
        summarize_paired_e3(panel, CONTRASTS, bootstrap_replicates=5)

    def test_duplicate_missing_and_misaligned_requests_are_rejected(self):
        mutations = [
            ("duplicate", lambda panel: panel["model_a"]["constrained"].append(
                copy.deepcopy(panel["model_a"]["constrained"][0]))),
            ("identities differ", lambda panel: panel["model_a"]["constrained"].pop()),
            ("identities differ", lambda panel: panel["model_b"]["constrained"][0].update(request_id="different")),
            ("clusters differ", lambda panel: panel["model_b"]["constrained"][0].update(subject_cluster="different")),
            ("clusters differ", lambda panel: panel["model_b"]["unconstrained"][0].update(note_cluster="different")),
            ("prompt hashes differ", lambda panel: panel["model_a"]["constrained"][0].update(prompt_sha256="c" * 64)),
        ]
        for message, mutate in mutations:
            with self.subTest(message=message):
                panel = _panel()
                mutate(panel)
                with self.assertRaisesRegex(ValueError, message):
                    summarize_paired_e3(panel, CONTRASTS, bootstrap_replicates=5)

    def test_changed_gold_across_models_rejected_even_if_scores_are_recomputed(self):
        panel = _panel()
        row = panel["model_b"]["unconstrained"][0]
        row["expected"] = copy.deepcopy(WRONG)
        row.update(_evaluation(row["raw_output"], row["expected"]))
        row["fence_normalized"] = _evaluation(row["raw_output"], row["expected"])
        with self.assertRaisesRegex(ValueError, "expected labels differ"):
            summarize_paired_e3(panel, CONTRASTS, bootstrap_replicates=5)

    def test_missing_invalid_and_stale_evaluations_rejected(self):
        changes = [
            ("fence_normalized", lambda row: row.pop("fence_normalized")),
            ("prompt_sha256", lambda row: row.pop("prompt_sha256")),
            ("hierarchical correctness", lambda row: row.update(hierarchical_joint_correct=False)),
            ("head predictions", lambda row: row["predictions"].update(detection="no")),
            ("must be boolean", lambda row: row.update(runtime_failed=0)),
            ("must be boolean", lambda row: row["parsed"].update(schema_valid=1)),
        ]
        for message, mutate in changes:
            with self.subTest(message=message):
                panel = _panel()
                mutate(panel["model_a"]["unconstrained"][0])
                with self.assertRaisesRegex(ValueError, message):
                    summarize_paired_e3(panel, CONTRASTS, bootstrap_replicates=5)

    def test_inconsistent_cluster_hierarchy_is_rejected(self):
        panel = _panel()
        for cells in panel.values():
            for rows in cells.values():
                rows[1]["subject_cluster"] = "another_subject"
        with self.assertRaisesRegex(ValueError, "consistent admission/subject"):
            summarize_paired_e3(panel, CONTRASTS, bootstrap_replicates=5)

    def test_incomplete_panel_and_invalid_contrasts_rejected(self):
        for panel in ({}, {"model_a": {}}, {"model_a": {"unconstrained": [], "constrained": []}}):
            with self.subTest(panel=panel), self.assertRaises(ValueError):
                summarize_paired_e3(panel, [], bootstrap_replicates=5)
        for contrasts in ([{"contrast_id": "x", "models": ["model_a", "missing"]}],
                          [{"contrast_id": "x", "models": ["model_a", "model_a"]}],
                          CONTRASTS + CONTRASTS):
            with self.subTest(contrasts=contrasts), self.assertRaises(ValueError):
                summarize_paired_e3(_panel(), contrasts, bootstrap_replicates=5)

    def test_bootstrap_controls_are_validated_and_single_subject_warns(self):
        for replicates in (0, -1, 2.5, True, float("inf")):
            with self.subTest(replicates=replicates), self.assertRaises(ValueError):
                summarize_paired_e3(_panel(), CONTRASTS, bootstrap_replicates=replicates)
        for seed in (-1, 2.5, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                summarize_paired_e3(_panel(), CONTRASTS, bootstrap_seed=seed)
        rows = [_row(0, GOLD), _row(1, WRONG)]
        result = summarize_paired_e3({"model_a": {"unconstrained": rows, "constrained": rows}}, [],
                                    bootstrap_replicates=1)
        self.assertIsNotNone(result["statistics"]["small_cluster_warning"])
        self.assertEqual(result["strict"]["cells"]["model_a"]["unconstrained"]
                         ["hierarchical_joint"]["subject_cluster_ci95"], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
