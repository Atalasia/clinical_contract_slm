from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import render_agreement_diagnostics_v0_13 as figure


def synthetic_fixture():
    payload = {"schema_version": "1.0", "analysis": "paired_compact_contract_factorial_followup",
               "data": {"model_count": 8, "requests_per_cell": 387, "vignette_count": 249,
                        "reference_state_counts": figure.REFERENCE_COUNTS,
                        "conditions": sorted(figure.SYNTHETIC_CONDITIONS)},
               "bootstrap": {"unit": "vignette_case_id", "replicates_requested": 10000,
                             "confidence": .95, "shared_draws_across_all_cells_and_models": True},
               "models": {}}
    for index, (model, _) in enumerate(figure.MODELS):
        delta = .01 * (index + 1)
        conditions = {}
        for condition in figure.SYNTHETIC_CONDITIONS:
            rate = .4 + delta if condition.startswith("compact") else .4
            conditions[condition] = {
                "n_requests": 387, "reference_state_counts": figure.REFERENCE_COUNTS,
                "schema_valid": {"count": 387, "denominator": 387, "rate": 1.0},
                "runtime_failure_count": 0, "length_truncation": {"count": 0},
                **{metric: {"estimate": rate} for metric in figure.SYNTHETIC_METRICS},
            }
        row = {"estimate": delta, "ci_low": delta - .01, "ci_high": delta + .01,
               "confidence": .95, "replicates_used": 10000, "replicates_excluded": 0}
        payload["models"][model] = {"conditions": conditions, "compact_minus_full": {
            "constrained": {metric: dict(row) for metric in figure.SYNTHETIC_METRICS}}}
    return payload


def note_fixture():
    clusters = {"notes": 150, "admissions": 130, "subjects": 130}
    payload = {"schema_version": "1.0", "primary_endpoint": "strict_hierarchical_joint_exact_match",
               "n_requests": 2288, "n_models": 8, "clusters": clusters,
               "conditions": ["unconstrained", "constrained"],
               "statistics": {"cluster_unit": "subject", "bootstrap_replicates": 10000,
                              "confidence_level": .95,
                              "paired_draws_shared_across_all_cells_and_scoring_views": True,
                              "decoding_difference_direction": "constrained minus unconstrained"}}
    for view in figure.NOTE_VIEWS:
        data = {"cells": {}, "within_model_decoding_effects": {}}
        for index, (model, _) in enumerate(figure.MODELS):
            counts = {"unconstrained": 1000, "constrained": 1020 + index}
            rates = {condition: count / 2288 for condition, count in counts.items()}
            delta = rates["constrained"] - rates["unconstrained"]
            data["cells"][model] = {
                condition: {"n": 2288, "runtime_completed": 2288, "clusters": clusters,
                            "hierarchical_joint": {"correct": count, "exact_match": rates[condition]}}
                for condition, count in counts.items()}
            data["within_model_decoding_effects"][model] = {
                "unconstrained_agreement": rates["unconstrained"],
                "constrained_agreement": rates["constrained"], "difference": delta,
                "subject_cluster_ci95": [delta - .001, delta + .001]}
        payload[view] = data
    return payload


class FigureInputTests(unittest.TestCase):
    def load_fixture(self, loader, payload):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "invented-aggregate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return loader(path)

    def test_all_eight_models_and_32_triplets_preserve_declared_order(self):
        synthetic = self.load_fixture(figure.load_synthetic, synthetic_fixture())
        notes = self.load_fixture(figure.load_notes, note_fixture())
        self.assertEqual(sum(len(series) for panel in (synthetic, notes) for series in panel.values()), 32)
        for metric in figure.SYNTHETIC_METRICS:
            for index, triplet in enumerate(synthetic[metric]):
                self.assertAlmostEqual(triplet[0], .01 * (index + 1))
        for view in figure.NOTE_VIEWS:
            for index, triplet in enumerate(notes[view]):
                self.assertAlmostEqual(triplet[0], (20 + index) / 2288)

    def test_synthetic_rejects_missing_model_wrong_denominator_and_orientation(self):
        model = figure.MODELS[0][0]
        mutations = [lambda p: p["models"].pop(model),
                     lambda p: p["data"].update(requests_per_cell=386),
                     lambda p: p["models"][model]["compact_minus_full"]["constrained"]["state_macro_agreement"].update(estimate=-.01),
                     lambda p: p["models"][model]["compact_minus_full"]["constrained"]["state_macro_agreement"].update(ci_low=.2),
                     lambda p: p["models"][model]["compact_minus_full"]["constrained"]["state_macro_agreement"].update(ci_high=float("nan"))]
        for mutation in mutations:
            payload = synthetic_fixture()
            mutation(payload)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load_fixture(figure.load_synthetic, payload)

    def test_notes_rejects_missing_model_count_rate_mismatch_and_orientation(self):
        model = figure.MODELS[0][0]
        mutations = [lambda p: p["strict"]["cells"].pop(model),
                     lambda p: p["clusters"].update(subjects=129),
                     lambda p: p["strict"]["cells"][model]["constrained"]["hierarchical_joint"].update(correct=1),
                     lambda p: p["strict"]["within_model_decoding_effects"][model].update(difference=-.01),
                     lambda p: p["statistics"].update(decoding_difference_direction="unconstrained minus constrained")]
        for mutation in mutations:
            payload = note_fixture()
            mutation(payload)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load_fixture(figure.load_notes, payload)

    def test_check_only_validates_without_plotting_or_writing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic, notes = root / "synthetic.json", root / "notes.json"
            synthetic.write_text(json.dumps(synthetic_fixture()), encoding="utf-8")
            notes.write_text(json.dumps(note_fixture()), encoding="utf-8")
            output = root / "must-not-exist.pdf"
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            argv = ["figure", "--synthetic-analysis", str(synthetic), "--notes-analysis", str(notes),
                    "--output", str(output), "--check-only"]
            with patch.object(sys, "argv", argv), patch.object(figure, "render") as render, \
                 patch.dict(sys.modules, {"matplotlib": None}), redirect_stdout(StringIO()) as out:
                figure.main()
            render.assert_not_called()
            self.assertIn("32 saved", out.getvalue())
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})


if __name__ == "__main__":
    unittest.main()
