from __future__ import annotations

import copy
import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cpg2 import slm_contract_ablation as ablation
from cpg2.slm_compact_contract import evaluate_contract_output


MODEL = "test_model"
CONDITION = "full_unconstrained"
RUNTIME = {"max_context_tokens": 2048, "max_output_tokens": {"typed_state": 192}}
PROVENANCE = {"manifest_id": MODEL, "experiment_id": "test_experiment",
              "runtime": RUNTIME, "prompt_tokenization": ablation.PROMPT_TOKENIZATION}
EXPECTED = {
    "request_1": {"case_id": "case_1", "domain": "test", "criterion_id": "criterion_1", "expected_state": "PRESENT_CURRENT"},
    "request_2": {"case_id": "case_2", "domain": "test", "criterion_id": "criterion_2", "expected_state": "ABSENT_EXPLICIT"},
}
ACCOUNTING = {request_id: {"prompt_sha256": "0" * 64, "prompt_tokens_preflight": 30,
                           "max_output_tokens": 192} for request_id in EXPECTED}


def _check_cell(folder, provenance=PROVENANCE, condition=CONDITION, expected=EXPECTED, **kwargs):
    return ablation._check_cell(folder, provenance, condition, expected,
                                prompt_accounting=kwargs.get("prompt_accounting", ACCOUNTING))


def _raw(criterion_id, state, contract, *, contradictory_abstention=False):
    compact = json.dumps({"criterion_id": criterion_id, "state": state,
                          "confidence": 0 if contradictory_abstention else 0.9,
                          "abstain": contradictory_abstention})
    if contract == "compact":
        return compact
    return evaluate_contract_output(compact, contract="compact", expected_criterion_id=criterion_id)["assembled_output"]


def _rows(condition=CONDITION):
    contract = condition.split("_", 1)[0]
    rows = []
    for request_id, metadata in EXPECTED.items():
        raw = _raw(metadata["criterion_id"], metadata["expected_state"], contract)
        evaluation = evaluate_contract_output(raw, contract=contract, expected_criterion_id=metadata["criterion_id"])
        rows.append({"request_id": request_id, **metadata,
                     "manifest_id": MODEL, "condition": condition,
                     "prompt_sha256": "0" * 64, "prompt_tokens": 30, "prompt_tokens_preflight": 30, "output_tokens": 17,
                     "raw_output": raw, **evaluation, "runtime_failed": False,
                     "finish_reason": "stop", "state_correct": True, "state_only_correct": True})
    return rows


def _cell(folder, *, rows=None, provenance=None, condition=CONDITION, audit_updates=None):
    responses = folder / "responses.jsonl"
    rows = _rows(condition) if rows is None else rows
    ablation._write_new(responses, rows, jsonl=True)
    audit = {
        "provenance": PROVENANCE if provenance is None else provenance,
        "condition": condition, "n": len(rows), "technical_completion": True,
        "responses_sha256": ablation._sha256_file(responses),
        "resolved_structured_backend": "xgrammar" if condition.endswith("_constrained") else None,
        "structured_schema_sha256": "0" * 64 if condition.endswith("_constrained") else None,
        "completed_at_utc": "2026-09-22T00:00:00+00:00", "inference_seconds": 0.01,
    }
    audit.update(audit_updates or {})
    ablation._write_new(folder / "audit.json", audit)
    return rows


class ContractAblationArtifactTests(unittest.TestCase):
    def test_exclusive_json_and_jsonl_writes_are_restricted_and_preserved(self):
        with TemporaryDirectory() as temporary:
            for name, jsonl in [("record.json", False), ("records.jsonl", True)]:
                with self.subTest(name=name):
                    destination = Path(temporary) / "nested" / name
                    value = [{"a": 1}, {"b": 2}] if jsonl else {"a": 1}
                    ablation._write_new(destination, value, jsonl=jsonl)
                    before = destination.read_bytes()
                    self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
                    with self.assertRaises(FileExistsError):
                        ablation._write_new(destination, {"overwrite": True}, jsonl=jsonl)
                    self.assertEqual(destination.read_bytes(), before)

    def test_missing_cell_is_distinguished_from_incomplete_cell(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.assertIsNone(_check_cell(folder))
            ablation._write_new(folder / "responses.jsonl", _rows(), jsonl=True)
            with self.assertRaisesRegex(ValueError, "incomplete artifacts"):
                _check_cell(folder)
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            ablation._write_new(folder / "audit.json", {})
            with self.assertRaisesRegex(ValueError, "incomplete artifacts"):
                _check_cell(folder)

    def test_completed_cell_is_returned_unchanged(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            expected_rows = _cell(folder)
            self.assertEqual(_check_cell(folder), expected_rows)

    def test_stale_provenance_wrong_condition_and_incompletion_are_rejected(self):
        changes = [{"provenance": {**PROVENANCE, "revision": "changed"}},
                   {"condition": "full_constrained"}, {"technical_completion": False},
                   {"provenance": None}]
        for change in changes:
            with self.subTest(change=change), TemporaryDirectory() as temporary:
                folder = Path(temporary)
                _cell(folder, audit_updates=change)
                with self.assertRaisesRegex(ValueError, "stale or incomplete"):
                    _check_cell(folder)

    def test_checksum_mismatch_is_rejected_before_reading_records(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _cell(folder, audit_updates={"responses_sha256": "changed"})
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                _check_cell(folder)

    def test_duplicate_missing_and_unexpected_requests_are_rejected(self):
        complete = _rows()
        unexpected = copy.deepcopy(complete)
        unexpected[1]["request_id"] = "not_in_blueprint"
        for rows in [complete + [complete[0]], complete[:1], unexpected]:
            with self.subTest(ids=[row["request_id"] for row in rows]), TemporaryDirectory() as temporary:
                folder = Path(temporary)
                _cell(folder, rows=rows)
                with self.assertRaisesRegex(ValueError, "duplicate/missing/unexpected"):
                    _check_cell(folder)

    def test_wrong_or_missing_blueprint_metadata_is_rejected(self):
        for key in EXPECTED["request_1"]:
            for missing in [False, True]:
                with self.subTest(key=key, missing=missing), TemporaryDirectory() as temporary:
                    folder = Path(temporary)
                    rows = _rows()
                    if missing:
                        del rows[0][key]
                    else:
                        rows[0][key] = "changed"
                    _cell(folder, rows=rows)
                    with self.assertRaisesRegex(ValueError, "response/blueprint mismatch"):
                        _check_cell(folder)

    def test_wrong_or_missing_model_condition_and_validation_are_rejected(self):
        for key in ["manifest_id", "condition", "raw_output", "parsed", "state_correct", "state_only_correct", "runtime_failed"]:
            with self.subTest(key=key), TemporaryDirectory() as temporary:
                folder = Path(temporary)
                rows = _rows()
                del rows[0][key]
                _cell(folder, rows=rows)
                with self.assertRaises(ValueError):
                    _check_cell(folder)

    def test_stored_validation_and_scores_are_recomputed_from_raw_text(self):
        mutations = [lambda row: row["parsed"].update(state="ABSENT_EXPLICIT"),
                     lambda row: row.update(state_correct=False),
                     lambda row: row.update(state_only_correct=False),
                     lambda row: row.update(assembled_output="{}")]
        for mutation in mutations:
            with self.subTest(mutation=mutation), TemporaryDirectory() as temporary:
                folder = Path(temporary)
                rows = _rows()
                mutation(rows[0])
                _cell(folder, rows=rows)
                with self.assertRaises(ValueError):
                    _check_cell(folder)

    def test_runtime_failure_cannot_be_resumed_as_completed_even_with_valid_checksum(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = _rows()
            rows[0].update(evaluate_contract_output(None, contract="full", expected_criterion_id=rows[0]["criterion_id"], runtime_failed=True))
            rows[0].update(raw_output=None, runtime_failed=True, state_correct=False, state_only_correct=False)
            _cell(folder, rows=rows)
            with self.assertRaisesRegex(ValueError, "scoring/completion"):
                _check_cell(folder)

    def test_resumed_cells_reject_changed_or_missing_token_accounting_with_valid_checksum(self):
        changes = [{"prompt_tokens": 31}, {"prompt_tokens": True}, {"prompt_tokens": None},
                   {"prompt_tokens_preflight": 31}, {"prompt_tokens_preflight": None},
                   {"prompt_sha256": "changed"}]
        for change in changes:
            with self.subTest(change=change), TemporaryDirectory() as temporary:
                folder = Path(temporary)
                rows = _rows()
                rows[0].update(change)
                _cell(folder, rows=rows)
                before = (folder / "responses.jsonl").read_bytes()
                with self.assertRaisesRegex(ValueError, "differs from preflight"):
                    _check_cell(folder)
                self.assertEqual((folder / "responses.jsonl").read_bytes(), before)

    def test_resumed_cell_validates_context_and_frozen_output_budget(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            provenance = {**PROVENANCE, "runtime": {**RUNTIME, "max_context_tokens": 221}}
            _cell(folder, provenance=provenance)
            with self.assertRaisesRegex(ValueError, "exceeds context"):
                _check_cell(folder, provenance=provenance)
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _cell(folder)
            changed_accounting = copy.deepcopy(ACCOUNTING)
            changed_accounting["request_1"]["max_output_tokens"] = 191
            with self.assertRaisesRegex(ValueError, "output budget"):
                _check_cell(folder, prompt_accounting=changed_accounting)

    def test_resumed_cell_requires_complete_preflight_request_mapping(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _cell(folder)
            with self.assertRaisesRegex(ValueError, "preflight requests"):
                _check_cell(folder, prompt_accounting={})

    def test_frozen_real_configuration_loads_without_model_loading(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / "outputs/local_restricted/slm_benchmark_v2/e1/dataset/manifest.json").exists():
            self.skipTest("integration check requires the locally retained frozen dataset manifest")
        experiment = ablation._experiment(root / ablation.DEFAULT_CONFIG)
        _, config, _, _, _, compact, _, panel = experiment
        self.assertEqual(tuple(config["conditions"]), ablation.CONDITIONS)
        self.assertEqual(len(panel), 8)
        self.assertEqual(config["expected_requests_per_cell"], 387)
        self.assertEqual(config["expected_cases"], 249)
        self.assertEqual(config["structured_backend"], "xgrammar")
        self.assertEqual(set(compact["output_schema"]["required"]), {"criterion_id", "state", "confidence", "abstain"})


class ContractAblationTokenAccountingTests(unittest.TestCase):
    def test_extra_bos_is_counted_without_changing_rendered_string_or_shared_renderer(self):
        for count, over_context in [(1857, True), (1856, False)]:
            with self.subTest(count=count):
                context = SimpleNamespace(manifest={"runtime": RUNTIME})
                prompt = "<bos>already-rendered-chat"
                row = {"request": {"request_id": "request_1"}, "rendered_prompt": prompt,
                       "prompt_tokens_preflight": count - 1, "max_output_tokens": 192,
                       "over_context": False, "response_prefix": ""}
                tokenizer = Mock()
                # Invented Gemma-like tokenizer: the rendered BOS is preserved,
                # and the historical string-input path prepends another BOS.
                tokenizer.encode.return_value = [2, 2] + [101] * (count - 2)
                with patch.object(ablation, "_render_e1_requests", return_value=([row], tokenizer)) as shared:
                    rendered, returned_tokenizer = ablation._render_ablation_requests(context)
                shared.assert_called_once_with(context)
                tokenizer.encode.assert_called_once_with(prompt, add_special_tokens=True)
                self.assertIs(returned_tokenizer, tokenizer)
                self.assertEqual(rendered[0]["rendered_prompt"], prompt)
                self.assertEqual(rendered[0]["prompt_tokens_preflight"], count)
                self.assertEqual(rendered[0]["over_context"], over_context)
                self.assertEqual(rendered[0]["response_prefix"], "")


class ContractAblationRunnerTests(unittest.TestCase):
    def _fixture(self, root):
        config = {"structured_backend": "xgrammar", "output_root": str(root),
                  "expected_requests_per_cell": 2, "expected_cases": 2}
        experiment = (root / "config.json", config, None, None, None, None, None, [MODEL])
        contexts = {name: SimpleNamespace(name=name, prompt={"output_schema": {"contract": name}},
                                         manifest={"runtime": RUNTIME})
                    for name in ["full", "compact"]}

        def render(context):
            return ([{"request": {"request_id": request_id, "criterion_id": metadata["criterion_id"]},
                      "rendered_prompt": f"{context.name}:{request_id}", "over_context": False,
                      "prompt_tokens_preflight": 3, "max_output_tokens": 192, "response_prefix": ""}
                     for request_id, metadata in EXPECTED.items()], None)

        ablation._write_new(root / "preflight.json", {
            "all_models_fit_context": True,
            "models": {MODEL: {"provenance": PROVENANCE, "contracts": {
                name: {"prompt_accounting": ablation._prompt_accounting(render(context)[0])}
                for name, context in contexts.items()}}},
        })
        return experiment, contexts, render

    def test_four_cells_use_matched_prompts_and_resumed_run_loads_no_engine(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, contexts, render = self._fixture(root)
            engine = Mock()

            def generate(prompts, params, *, use_tqdm):
                self.assertFalse(use_tqdm)
                contract, _ = params.split("_", 1)
                outputs = []
                for prompt in prompts:
                    request_id = prompt.split(":", 1)[1]
                    metadata = EXPECTED[request_id]
                    raw = _raw(metadata["criterion_id"], metadata["expected_state"], contract,
                               contradictory_abstention=params == "compact_constrained" and request_id == "request_2")
                    choice = SimpleNamespace(text=raw, token_ids=[1, 2, 3, 4], finish_reason="stop")
                    outputs.append(SimpleNamespace(outputs=[choice], prompt_token_ids=[1, 2, 3]))
                return outputs

            engine.generate.side_effect = generate
            with patch.object(ablation, "_experiment", return_value=experiment), \
                 patch.object(ablation, "_contexts", return_value=contexts), \
                 patch.object(ablation, "_provenance", return_value=PROVENANCE), \
                 patch.object(ablation, "_expected_rows", return_value=EXPECTED), \
                 patch.object(ablation, "_render_ablation_requests", side_effect=render) as renderer, \
                 patch.object(ablation, "_load_engine", return_value=engine) as loader, \
                 patch("torch.cuda.get_device_name", return_value="Mock GPU"), \
                 patch.object(ablation, "_sampling_params", side_effect=lambda context, budget: context.name + "_unconstrained") as unstructured, \
                 patch.object(ablation, "_structured_sampling_params", side_effect=lambda context: context.name + "_constrained") as structured:
                ablation.run_model(manifest_id=MODEL, output_root=root)
                loader.assert_called_once_with(contexts["full"], "xgrammar")
                self.assertEqual(engine.generate.call_count, 4)
                self.assertEqual(renderer.call_count, 2)
                calls = engine.generate.call_args_list
                self.assertEqual(calls[0].args[0], calls[1].args[0])
                self.assertEqual(calls[2].args[0], calls[3].args[0])
                self.assertNotEqual(calls[0].args[0], calls[2].args[0])
                self.assertEqual([call.args[1] for call in calls], list(ablation.CONDITIONS))
                self.assertTrue(all(call.kwargs == {"use_tqdm": False} for call in calls))
                for condition in ablation.CONDITIONS:
                    rows = _check_cell(root / "results" / MODEL / condition, condition=condition,
                                       prompt_accounting=ablation._prompt_accounting(render(contexts[condition.split("_", 1)[0]])[0]))
                    self.assertEqual(len(rows), 2)
                    for row in rows:
                        self.assertEqual(row["prompt_tokens"], 3)
                        self.assertEqual(row["prompt_tokens_preflight"], 3)
                        self.assertEqual(row["output_tokens"], 4)
                        self.assertTrue(row["state_only_correct"])
                        self.assertEqual(row["state_correct"], not (condition == "compact_constrained" and row["request_id"] == "request_2"))
                        if condition.startswith("compact_"):
                            self.assertEqual(len(json.loads(row["raw_output"])), 4)
                            self.assertEqual(len(json.loads(row["assembled_output"])), 8)
                        else:
                            self.assertEqual(row["raw_output"], row["assembled_output"])
                for mock in [loader, renderer, engine.generate, unstructured, structured]:
                    mock.reset_mock()
                ablation.run_model(manifest_id=MODEL, output_root=root)
                self.assertEqual(renderer.call_count, 2)
                for mock in [loader, engine.generate, unstructured, structured]:
                    mock.assert_not_called()

    def test_stale_preflight_is_rejected_before_engine_load(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, contexts, _ = self._fixture(root)
            with patch.object(ablation, "_experiment", return_value=experiment), \
                 patch.object(ablation, "_contexts", return_value=contexts), \
                 patch.object(ablation, "_provenance", return_value={**PROVENANCE, "changed": True}), \
                 patch.object(ablation, "_load_engine") as loader:
                with self.assertRaisesRegex(ValueError, "preflight"):
                    ablation.run_model(manifest_id=MODEL, output_root=root)
                loader.assert_not_called()

    def test_legacy_preflight_without_tokenization_policy_is_not_rebound(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_provenance = {key: value for key, value in PROVENANCE.items() if key != "prompt_tokenization"}
            ablation._write_new(root / "preflight.json", {
                "all_models_fit_context": True,
                "models": {MODEL: {"provenance": legacy_provenance}},
            })
            before = (root / "preflight.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "fresh output root"):
                ablation._preflight_model(root, MODEL, legacy_provenance)
            self.assertEqual((root / "preflight.json").read_bytes(), before)

    def test_preflight_persists_request_level_accounting_and_does_not_replace_old_audit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, contexts, render = self._fixture(root)
            fresh_root = root / "new"
            with patch.object(ablation, "_experiment", return_value=experiment), \
                 patch.object(ablation, "_contexts", return_value=contexts), \
                 patch.object(ablation, "_provenance", return_value=PROVENANCE), \
                 patch.object(ablation, "_expected_rows", return_value=EXPECTED), \
                 patch.object(ablation, "_render_ablation_requests", side_effect=render):
                report = ablation.preflight(output_root=fresh_root)
                self.assertEqual(report["models"][MODEL]["provenance"]["prompt_tokenization"],
                                 {"input": "rendered_string", "add_special_tokens": True})
                for contract in contexts:
                    cell = report["models"][MODEL]["contracts"][contract]
                    self.assertEqual(cell["prompt_tokens_max"], 3)
                    self.assertEqual(cell["prompt_accounting"], ablation._prompt_accounting(render(contexts[contract])[0]))
                self.assertEqual(ablation.preflight(output_root=fresh_root), report)
                before = (root / "preflight.json").read_bytes()
                with self.assertRaisesRegex(ValueError, "existing preflight differs"):
                    ablation.preflight(output_root=root)
                self.assertEqual((root / "preflight.json").read_bytes(), before)

    def test_re_rendered_count_or_hash_must_match_preflight_before_engine_load(self):
        for key, value in [("prompt_tokens_preflight", 4), ("rendered_prompt", "changed")]:
            with self.subTest(key=key), TemporaryDirectory() as temporary:
                root = Path(temporary)
                experiment, contexts, render = self._fixture(root)

                def changed(context):
                    rows, tokenizer = render(context)
                    rows[0][key] = value
                    return rows, tokenizer

                with patch.object(ablation, "_experiment", return_value=experiment), \
                     patch.object(ablation, "_contexts", return_value=contexts), \
                     patch.object(ablation, "_provenance", return_value=PROVENANCE), \
                     patch.object(ablation, "_expected_rows", return_value=EXPECTED), \
                     patch.object(ablation, "_render_ablation_requests", side_effect=changed), \
                     patch.object(ablation, "_load_engine") as loader:
                    with self.assertRaisesRegex(ValueError, "differ from preflight"):
                        ablation.run_model(manifest_id=MODEL, output_root=root)
                    loader.assert_not_called()

    def test_runtime_token_count_mismatch_or_missing_ids_leaves_no_completed_cell(self):
        for token_ids in [[2, 2, 101, 102], None]:
            with self.subTest(token_ids=token_ids), TemporaryDirectory() as temporary:
                root = Path(temporary)
                experiment, contexts, render = self._fixture(root)
                engine = Mock()
                engine.generate.return_value = [SimpleNamespace(
                    outputs=[SimpleNamespace(text=_raw(metadata["criterion_id"], metadata["expected_state"], "full"),
                                             token_ids=[101], finish_reason="stop")],
                    prompt_token_ids=token_ids) for metadata in EXPECTED.values()]
                with patch.object(ablation, "_experiment", return_value=experiment), \
                     patch.object(ablation, "_contexts", return_value=contexts), \
                     patch.object(ablation, "_provenance", return_value=PROVENANCE), \
                     patch.object(ablation, "_expected_rows", return_value=EXPECTED), \
                     patch.object(ablation, "_render_ablation_requests", side_effect=render), \
                     patch.object(ablation, "_load_engine", return_value=engine), \
                     patch("torch.cuda.get_device_name", return_value="Mock GPU"), \
                     patch.object(ablation, "_sampling_params", return_value="mock"):
                    with self.assertRaisesRegex(ValueError, "prompt token count differs"):
                        ablation.run_model(manifest_id=MODEL, output_root=root)
                    engine.generate.assert_called_once_with(
                        [row["rendered_prompt"] for row in render(contexts["full"])[0]], "mock", use_tqdm=False)
                    self.assertFalse((root / "results").exists())

    def test_runtime_prompt_order_mismatch_is_rejected_even_when_counts_match(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, contexts, render = self._fixture(root)
            engine = Mock()
            engine.generate.return_value = [SimpleNamespace(
                outputs=[], prompt_token_ids=[1, 2, 3], prompt="different request") for _ in EXPECTED]
            with patch.object(ablation, "_experiment", return_value=experiment), \
                 patch.object(ablation, "_contexts", return_value=contexts), \
                 patch.object(ablation, "_provenance", return_value=PROVENANCE), \
                 patch.object(ablation, "_expected_rows", return_value=EXPECTED), \
                 patch.object(ablation, "_render_ablation_requests", side_effect=render), \
                 patch.object(ablation, "_load_engine", return_value=engine), \
                 patch("torch.cuda.get_device_name", return_value="Mock GPU"), \
                 patch.object(ablation, "_sampling_params", return_value="mock"):
                with self.assertRaisesRegex(RuntimeError, "prompt different"):
                    ablation.run_model(manifest_id=MODEL, output_root=root)
                self.assertFalse((root / "results").exists())

    def test_aggregate_checks_saved_prompt_counts_without_tokenizer_or_engine(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, contexts, render = self._fixture(root)
            config = {**experiment[1], "statistics": {"bootstrap_replicates": 10, "bootstrap_seed": 2026}}
            experiment = (experiment[0], config, *experiment[2:6], {"contrasts": []}, [MODEL])
            for condition in ablation.CONDITIONS:
                rows = _rows(condition)
                accounting = ablation._prompt_accounting(render(contexts[condition.split("_", 1)[0]])[0])
                for row in rows:
                    row.update({key: value for key, value in accounting[row["request_id"]].items()
                                if key != "max_output_tokens"})
                    row["prompt_tokens"] = 3
                _cell(root / "results" / MODEL / condition, rows=rows, condition=condition)
            with patch.object(ablation, "_experiment", return_value=experiment), \
                 patch.object(ablation, "_contexts", return_value=contexts), \
                 patch.object(ablation, "_provenance", return_value=PROVENANCE), \
                 patch.object(ablation, "_expected_rows", return_value=EXPECTED), \
                 patch.object(ablation, "_render_ablation_requests") as renderer, \
                 patch.object(ablation, "_load_engine") as loader, \
                 patch("cpg2.slm_contract_analysis.summarize_ablation", return_value={"checked": True}) as summary:
                result = ablation.aggregate(output_root=root)
                self.assertTrue(result["checked"])
                self.assertEqual(set(summary.call_args.args[0][MODEL]), set(ablation.CONDITIONS))
                renderer.assert_not_called()
                loader.assert_not_called()

    def test_over_context_runtime_requests_are_rejected_before_engine_load(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, contexts, render = self._fixture(root)

            def over_context(context):
                rows, tokenizer = render(context)
                rows[0]["over_context"] = True
                return rows, tokenizer

            with patch.object(ablation, "_experiment", return_value=experiment), \
                 patch.object(ablation, "_contexts", return_value=contexts), \
                 patch.object(ablation, "_provenance", return_value=PROVENANCE), \
                 patch.object(ablation, "_expected_rows", return_value=EXPECTED), \
                 patch.object(ablation, "_render_ablation_requests", side_effect=over_context), \
                 patch.object(ablation, "_load_engine") as loader:
                with self.assertRaisesRegex(ValueError, "over-context"):
                    ablation.run_model(manifest_id=MODEL, output_root=root)
                loader.assert_not_called()

    def test_wrong_response_count_leaves_no_completed_cell(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, contexts, render = self._fixture(root)
            engine = Mock()
            engine.generate.return_value = []
            with patch.object(ablation, "_experiment", return_value=experiment), \
                 patch.object(ablation, "_contexts", return_value=contexts), \
                 patch.object(ablation, "_provenance", return_value=PROVENANCE), \
                 patch.object(ablation, "_expected_rows", return_value=EXPECTED), \
                 patch.object(ablation, "_render_ablation_requests", side_effect=render), \
                 patch.object(ablation, "_load_engine", return_value=engine), \
                 patch("torch.cuda.get_device_name", return_value="Mock GPU"), \
                 patch.object(ablation, "_sampling_params", return_value="mock"):
                with self.assertRaisesRegex(RuntimeError, "wrong number"):
                    ablation.run_model(manifest_id=MODEL, output_root=root)
                self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
