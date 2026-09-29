from __future__ import annotations

import copy
from contextlib import ExitStack
import json
from pathlib import Path
import stat
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cpg2 import slm_e3_contract as paired
from cpg2.slm_e3 import FAILURE_LABEL, _E3Context


MODEL = "test_model"
EXPECTED = {"detection": "yes", "encounter": "yes", "negation": "unsure"}
PROVENANCE = {"manifest_id": MODEL, "experiment_id": "test_experiment"}


def _rendered(count=3):
    return [
        {"request": {"request_id": f"request_{index}", "expected": EXPECTED.copy(),
                     "note_cluster": f"note_{index}", "subject_cluster": f"subject_{index // 2}",
                     "admission_cluster": f"admission_{index // 2}",
                     "semantic_group": "test", "context_mode": "complete_note"},
         "rendered_prompt": f"synthetic prompt {index}",
         "prompt_sha256": str(index) * 64, "prompt_tokens_preflight": 3,
         "prompt_token_ids_sha256": paired.config_sha256([1, 2, 3])}
        for index in range(count)
    ]


def _generated(prompts, params=None, *, use_tqdm=False, tokenization_kwargs=None):
    return [SimpleNamespace(prompt=prompt, prompt_token_ids=[1, 2, 3], outputs=[
        SimpleNamespace(text=json.dumps(EXPECTED), token_ids=[1, 2, 3, 4], finish_reason="stop")
    ]) for prompt in prompts]


def _batch_value(rendered, condition="unconstrained", index=0):
    rows = [paired._record(item, output, manifest_id=MODEL, condition=condition)
            for item, output in zip(rendered, _generated([item["rendered_prompt"] for item in rendered]))]
    return {"provenance": PROVENANCE, "condition": condition, "batch_index": index,
            "technical_completion": True, "rows": rows, "rows_sha256": paired.config_sha256(rows),
            "hardware": {"gpu_name": "Mock GPU", "cuda_runtime": "mock"},
            "completed_at_utc": "2026-09-25T00:00:00+00:00", "inference_seconds": 0.01}


class E3ContractEvaluationTests(unittest.TestCase):
    def test_native_unsure_is_a_correct_label_not_model_abstention(self):
        result = paired.evaluate_output(json.dumps(EXPECTED), EXPECTED)
        self.assertTrue(result["hierarchical_joint_correct"])
        self.assertTrue(result["parsed"]["schema_valid"])
        self.assertTrue(result["parsed"]["logical_valid"])
        self.assertFalse(result["parsed"]["abstained"])
        self.assertEqual(result["predictions"], EXPECTED)
        self.assertFalse(result["fence_removed"])

    def test_abstention_does_not_match_native_unsure(self):
        result = paired.evaluate_output(json.dumps({**EXPECTED, "negation": "abstain"}), EXPECTED)
        self.assertTrue(result["parsed"]["logical_valid"])
        self.assertTrue(result["parsed"]["abstained"])
        self.assertFalse(result["hierarchical_joint_correct"])
        self.assertEqual(result["predictions"]["detection"], "yes")
        self.assertEqual(result["predictions"]["negation"], FAILURE_LABEL)

    def test_structurally_valid_inconsistent_hierarchy_still_fails(self):
        result = paired.evaluate_output(json.dumps({**EXPECTED, "detection": "no"}), EXPECTED)
        self.assertTrue(result["parsed"]["schema_valid"])
        self.assertFalse(result["parsed"]["logical_valid"])
        self.assertFalse(result["hierarchical_joint_correct"])
        self.assertEqual(set(result["predictions"].values()), {FAILURE_LABEL})

    def test_detection_no_uses_native_hierarchy(self):
        expected = {"detection": "no", "encounter": "not_applicable", "negation": "not_applicable"}
        result = paired.evaluate_output(json.dumps(expected), expected)
        self.assertTrue(result["hierarchical_joint_correct"])

    def test_fence_normalization_is_secondary_only(self):
        for language in ["json", ""]:
            with self.subTest(language=language):
                raw = "```" + language + "\n" + json.dumps(EXPECTED) + "\n```"
                result = paired.evaluate_output(raw, EXPECTED)
                self.assertTrue(result["fence_removed"])
                self.assertFalse(result["parsed"]["schema_valid"])
                self.assertFalse(result["hierarchical_joint_correct"])
                self.assertTrue(result["fence_normalized"]["hierarchical_joint_correct"])
                self.assertEqual(result["fence_normalized"]["predictions"], EXPECTED)

    def test_fence_normalization_does_not_extract_json_from_prose(self):
        valid = json.dumps(EXPECTED)
        for raw in ["Here is the answer: " + valid,
                    "```json\n" + valid + "\n```\nExplanation",
                    "Explanation\n```json\n" + valid + "\n```",
                    "```json\n" + valid,
                    valid + "\n" + valid]:
            with self.subTest(raw=raw):
                result = paired.evaluate_output(raw, EXPECTED)
                self.assertFalse(result["fence_removed"])
                self.assertFalse(result["hierarchical_joint_correct"])
                self.assertFalse(result["fence_normalized"]["hierarchical_joint_correct"])

    def test_extra_fields_are_not_repaired(self):
        result = paired.evaluate_output(json.dumps({**EXPECTED, "explanation": "extra"}), EXPECTED)
        self.assertFalse(result["parsed"]["schema_valid"])
        self.assertFalse(result["fence_normalized"]["hierarchical_joint_correct"])

    def test_runtime_failure_cannot_score_valid_looking_output(self):
        result = paired.evaluate_output(json.dumps(EXPECTED), EXPECTED, runtime_failed=True)
        self.assertEqual(result["parsed"]["failure"], "runtime_failure")
        self.assertFalse(result["hierarchical_joint_correct"])
        self.assertFalse(result["fence_normalized"]["hierarchical_joint_correct"])


class E3ContractArtifactTests(unittest.TestCase):
    def test_exclusive_json_and_jsonl_writes_are_restricted_and_preserved(self):
        with TemporaryDirectory() as temporary:
            for name, jsonl in [("record.json", False), ("records.jsonl", True)]:
                with self.subTest(name=name):
                    destination = Path(temporary) / "nested" / name
                    value = [{"a": 1}, {"b": 2}] if jsonl else {"a": 1}
                    paired._write_new(destination, value, jsonl=jsonl)
                    before = destination.read_bytes()
                    self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
                    with self.assertRaises(FileExistsError):
                        paired._write_new(destination, {"overwrite": True}, jsonl=jsonl)
                    self.assertEqual(destination.read_bytes(), before)

    def test_failed_serialization_leaves_no_final_artifact(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            with self.assertRaises((TypeError, ValueError)):
                paired._write_new(path, {"invalid": object()})
            self.assertFalse(path.exists())

    def test_matching_write_is_idempotent_but_rejects_changed_content(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            paired._write_matching(path, {"a": 1})
            original = path.read_bytes()
            paired._write_matching(path, {"a": 1})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                paired._write_matching(path, {"a": 2})
            self.assertEqual(path.read_bytes(), original)


class E3ContractSamplingTests(unittest.TestCase):
    def test_schema_is_only_sampler_difference_and_does_not_add_hierarchy_rules(self):
        schema = {
            "type": "object",
            "properties": {
                "detection": {"enum": ["yes", "no", "abstain"]},
                "encounter": {"enum": ["yes", "no", "not_applicable", "abstain"]},
                "negation": {"enum": ["yes", "no", "unsure", "not_applicable", "abstain"]},
            },
            "required": ["detection", "encounter", "negation"],
            "additionalProperties": False,
        }
        context = SimpleNamespace(
            config={"generation": {"temperature": 0, "max_output_tokens": 96, "seed": 2026}},
            prompt={"output_schema": schema},
        )
        sampling = Mock(side_effect=lambda **kwargs: kwargs)
        structured = Mock(side_effect=lambda **kwargs: kwargs)
        with patch.dict(sys.modules, {
            "vllm": SimpleNamespace(SamplingParams=sampling),
            "vllm.sampling_params": SimpleNamespace(StructuredOutputsParams=structured),
        }):
            unstructured = paired._sampling_params(context, "unconstrained")
            constrained = paired._sampling_params(context, "constrained")
        self.assertNotIn("structured_outputs", unstructured)
        self.assertEqual(constrained.pop("structured_outputs")["json"], schema)
        self.assertEqual(unstructured, constrained)
        self.assertEqual(unstructured["max_tokens"], 96)
        self.assertEqual(unstructured["temperature"], 0)
        self.assertEqual(unstructured["seed"], 2026)
        self.assertEqual(structured.call_count, 1)

    def test_engine_pins_backend_and_disables_prefix_cache(self):
        context = SimpleNamespace(
            config={"generation": {"dtype": "bfloat16", "gpu_memory_utilization": 0.85,
                                   "max_num_seqs": 8, "trust_remote_code": False, "seed": 2026}},
            manifest={"model": {"local_snapshot": "/synthetic/model"}},
        )
        engine = Mock()
        engine.llm_engine.vllm_config.structured_outputs_config.backend = "xgrammar"
        loader = Mock(return_value=engine)
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(LLM=loader)}), \
             patch.object(paired, "_multimodal_limits", return_value=None):
            self.assertIs(paired._load_engine(context, "xgrammar"), engine)
        self.assertFalse(loader.call_args.kwargs["enable_prefix_caching"])
        self.assertEqual(loader.call_args.kwargs["structured_outputs_config"], {"backend": "xgrammar"})
        self.assertEqual(loader.call_args.kwargs["max_model_len"], 2048)


class E3ContractCheckpointTests(unittest.TestCase):
    def test_absent_checkpoint_and_valid_checkpoint(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "batch.json"
            rendered = _rendered()
            self.assertIsNone(paired._check_batch(path, PROVENANCE, "unconstrained", 0, rendered))
            batch = _batch_value(rendered)
            paired._write_new(path, batch)
            self.assertEqual(paired._check_batch(path, PROVENANCE, "unconstrained", 0, rendered), batch)

    def test_stale_metadata_and_checksum_are_rejected(self):
        changes = [{"provenance": {}}, {"condition": "constrained"}, {"batch_index": 1},
                   {"technical_completion": False}, {"rows_sha256": "stale"}]
        for change in changes:
            with self.subTest(change=change), TemporaryDirectory() as temporary:
                path = Path(temporary) / "batch.json"
                rendered = _rendered()
                paired._write_new(path, {**_batch_value(rendered), **change})
                with self.assertRaises(ValueError):
                    paired._check_batch(path, PROVENANCE, "unconstrained", 0, rendered)

    def test_metadata_and_validation_are_rechecked_even_with_valid_checksum(self):
        mutations = [
            lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows.pop(),
            lambda rows: rows.reverse(),
            lambda rows: rows[0].update(request_id="unexpected"),
            lambda rows: rows[0].update(subject_cluster="changed"),
            lambda rows: rows[0].update(expected={**EXPECTED, "negation": "yes"}),
            lambda rows: rows[0].update(condition="constrained"),
            lambda rows: rows[0].update(manifest_id="different_model"),
            lambda rows: rows[0].update(prompt_sha256="changed"),
            lambda rows: rows[0].update(prompt_token_ids_sha256="changed"),
            lambda rows: rows[0].update(prompt_tokens_preflight=999),
            lambda rows: rows[0].update(hierarchical_joint_correct=False),
            lambda rows: rows[0]["parsed"].update(logical_valid=False),
            lambda rows: rows[0]["fence_normalized"].update(hierarchical_joint_correct=False),
            lambda rows: rows[0].update(runtime_failed=True),
            lambda rows: rows[0].update(raw_output=None),
            lambda rows: rows[0].update(prompt_tokens=-1),
            lambda rows: rows[0].update(prompt_tokens=999),
            lambda rows: rows[0].update(output_tokens=True),
            lambda rows: rows[0].pop("finish_reason"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), TemporaryDirectory() as temporary:
                path = Path(temporary) / "batch.json"
                rendered = _rendered()
                batch = _batch_value(rendered)
                mutate(batch["rows"])
                batch["rows_sha256"] = paired.config_sha256(batch["rows"])
                paired._write_new(path, batch)
                with self.assertRaises(ValueError):
                    paired._check_batch(path, PROVENANCE, "unconstrained", 0, rendered)

    def test_unexpected_batch_file_is_not_silently_ignored(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            paired._write_new(folder / "batches/batch_00999.json", {})
            with self.assertRaisesRegex(ValueError, "unexpected checkpoint"):
                paired._read_batches(folder, paired._batch_specs(_rendered(), 2), PROVENANCE, "unconstrained")

    def test_insecure_checkpoint_permissions_are_rejected(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "batch.json"
            rendered = _rendered()
            paired._write_new(path, _batch_value(rendered))
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "permissions"):
                paired._check_batch(path, PROVENANCE, "unconstrained", 0, rendered)


class E3ContractRunnerTests(unittest.TestCase):
    def _fixture(self, root):
        rendered = _rendered()
        experiment = SimpleNamespace(panel=(MODEL,), config={"checkpoint_batch_size": 2, "structured_backend": "xgrammar"})
        context = SimpleNamespace()
        model_preflight = {"provenance": PROVENANCE}
        paired._write_new(root / "preflight.json", {
            "all_models_fit_context": True, "experiment_config_sha256": paired.config_sha256(experiment.config),
            "models": {MODEL: model_preflight},
        })
        engine = Mock()
        engine.generate.side_effect = _generated
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(paired, "_experiment", return_value=experiment))
        stack.enter_context(patch.object(paired, "_output_root", return_value=root))
        stack.enter_context(patch.object(paired, "_context", return_value=context))
        stack.enter_context(patch.object(paired, "_render_requests", return_value=rendered))
        stack.enter_context(patch.object(paired, "_model_preflight", return_value=model_preflight))
        loader = stack.enter_context(patch.object(paired, "_load_engine", return_value=engine))
        sampler = stack.enter_context(patch.object(paired, "_sampling_params", side_effect=lambda context, condition: condition))
        stack.enter_context(patch.object(paired, "_hardware", return_value={"gpu_name": "Mock GPU", "cuda_runtime": "mock"}))
        return rendered, experiment, engine, loader, sampler

    def test_identical_prompts_across_conditions_and_completed_resume_loads_no_engine(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rendered, _, engine, loader, sampler = self._fixture(root)
            paired.run_model(manifest_id=MODEL, output_root=root)
            self.assertEqual(engine.generate.call_count, 4)
            calls = engine.generate.call_args_list
            self.assertEqual(calls[0].args[0], calls[2].args[0])
            self.assertEqual(calls[1].args[0], calls[3].args[0])
            self.assertEqual([call.args[1] for call in calls], ["unconstrained"] * 2 + ["constrained"] * 2)
            self.assertTrue(all(call.kwargs["tokenization_kwargs"] == {"add_special_tokens": False} for call in calls))
            for condition in paired.CONDITIONS:
                rows = list(paired.read_jsonl(root / "results" / MODEL / condition / "responses.jsonl"))
                self.assertEqual([row["request_id"] for row in rows], [item["request"]["request_id"] for item in rendered])
                self.assertTrue(all(row["hierarchical_joint_correct"] for row in rows))
                self.assertTrue(all(row["output_tokens"] == 4 for row in rows))
            loader.reset_mock()
            sampler.reset_mock()
            engine.generate.reset_mock()
            paired.run_model(manifest_id=MODEL, output_root=root)
            loader.assert_not_called()
            sampler.assert_not_called()
            engine.generate.assert_not_called()

    def test_partial_resume_skips_completed_batch_and_preserves_its_bytes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rendered, _, engine, _, _ = self._fixture(root)
            checkpoint = root / "results" / MODEL / "unconstrained/batches/batch_00000.json"
            paired._write_new(checkpoint, _batch_value(rendered[:2]))
            before = checkpoint.read_bytes()
            paired.run_model(manifest_id=MODEL, output_root=root)
            self.assertEqual(engine.generate.call_count, 3)
            self.assertEqual(engine.generate.call_args_list[0].args,
                             ([rendered[2]["rendered_prompt"]], "unconstrained"))
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_stale_preflight_fails_before_engine_load(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, loader, _ = self._fixture(root)
            with patch.object(paired, "_model_preflight", return_value={"provenance": {"changed": True}}):
                with self.assertRaisesRegex(ValueError, "preflight"):
                    paired.run_model(manifest_id=MODEL, output_root=root)
            loader.assert_not_called()

    def test_wrong_output_count_does_not_publish_completed_batch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, engine, _, _ = self._fixture(root)
            engine.generate.side_effect = lambda *args, **kwargs: []
            with self.assertRaisesRegex(RuntimeError, "wrong number"):
                paired.run_model(manifest_id=MODEL, output_root=root)
            self.assertFalse((root / "results").exists())

    def test_prompt_order_mismatch_does_not_publish_batch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, engine, _, _ = self._fixture(root)
            engine.generate.side_effect = lambda prompts, *args, **kwargs: _generated(list(reversed(prompts)))
            with self.assertRaisesRegex(RuntimeError, "different prompt"):
                paired.run_model(manifest_id=MODEL, output_root=root)
            self.assertFalse((root / "results").exists())

    def test_engine_tokenization_mismatch_does_not_publish_batch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, engine, _, _ = self._fixture(root)

            def wrong_tokens(prompts, *args, **kwargs):
                generated = _generated(prompts)
                generated[0].prompt_token_ids = [9, 2, 3]
                return generated

            engine.generate.side_effect = wrong_tokens
            with self.assertRaisesRegex(RuntimeError, "token IDs"):
                paired.run_model(manifest_id=MODEL, output_root=root)
            self.assertFalse((root / "results").exists())

    def test_invalid_json_is_completed_failure_not_retried(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, engine, loader, _ = self._fixture(root)

            def invalid_output(prompts, *args, **kwargs):
                generated = _generated(prompts)
                for result in generated:
                    result.outputs[0].text = "not JSON"
                return generated

            engine.generate.side_effect = invalid_output
            paired.run_model(manifest_id=MODEL, output_root=root)
            for condition in paired.CONDITIONS:
                rows = list(paired.read_jsonl(root / "results" / MODEL / condition / "responses.jsonl"))
                self.assertTrue(all(not row["runtime_failed"] for row in rows))
                self.assertTrue(all(not row["hierarchical_joint_correct"] for row in rows))
            loader.reset_mock()
            paired.run_model(manifest_id=MODEL, output_root=root)
            loader.assert_not_called()

    def test_runtime_failure_is_preserved_but_never_resumed_as_complete(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, engine, loader, _ = self._fixture(root)

            def missing_output(prompts, *args, **kwargs):
                generated = _generated(prompts)
                generated[0].outputs = []
                return generated

            engine.generate.side_effect = missing_output
            with self.assertRaisesRegex(RuntimeError, "missing engine output"):
                paired.run_model(manifest_id=MODEL, output_root=root)
            checkpoint = root / "results" / MODEL / "unconstrained/batches/batch_00000.json"
            self.assertFalse(checkpoint.exists())
            attempts = list((root / "results" / MODEL / "unconstrained/failed_attempts").glob("*.json"))
            self.assertEqual(len(attempts), 1)
            before = attempts[0].read_bytes()
            batch = json.loads(before)
            self.assertFalse(batch["technical_completion"])
            self.assertTrue(batch["rows"][0]["runtime_failed"])
            loader.reset_mock()
            engine.generate.reset_mock()
            engine.generate.side_effect = _generated
            paired.run_model(manifest_id=MODEL, output_root=root)
            loader.assert_called_once()
            self.assertEqual(engine.generate.call_count, 4)
            self.assertEqual(attempts[0].read_bytes(), before)
            self.assertTrue(checkpoint.exists())

    def test_engine_failure_after_first_batch_can_resume_remaining_work(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, engine, loader, _ = self._fixture(root)
            calls = 0

            def interrupt(prompts, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("mock interrupted engine")
                return _generated(prompts)

            engine.generate.side_effect = interrupt
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                paired.run_model(manifest_id=MODEL, output_root=root)
            checkpoint = root / "results" / MODEL / "unconstrained/batches/batch_00000.json"
            before = checkpoint.read_bytes()
            engine.generate.reset_mock()
            engine.generate.side_effect = _generated
            paired.run_model(manifest_id=MODEL, output_root=root)
            self.assertEqual(engine.generate.call_count, 3)
            self.assertEqual(checkpoint.read_bytes(), before)


class E3ContractContextTests(unittest.TestCase):
    def _context(self, wrapper):
        return _E3Context(
            config_path=Path("config.json"), benchmark_path=Path("benchmark.json"),
            manifest_path=Path("manifest.json"), wrapper_path=Path("wrapper.json"),
            prompt_path=Path("prompt.json"), requests_path=Path("requests.jsonl"),
            config={"generation": {"dtype": "bfloat16", "temperature": 0, "seed": 2026,
                                   "max_output_tokens": 96},
                    "context_policy": {"max_context_tokens": 2048}},
            benchmark={}, manifest={"runtime": {"quantization": None}}, wrapper=wrapper, prompt={})

    def test_prefill_and_continued_assistant_wrappers_are_rejected(self):
        experiment = SimpleNamespace(primary=Path("primary.json"), dataset=Path("dataset.json"),
                                     panel=(MODEL,), config={"wrapper_overrides": {}})
        wrappers = [{"assistant_prefill": "{"}, {"continue_final_message": True},
                    {"messages": [{"role": "assistant", "content": "{"}]}]
        for wrapper in wrappers:
            with self.subTest(wrapper=wrapper), patch.object(paired, "_load_model_context", return_value=self._context(wrapper)):
                with self.assertRaisesRegex(ValueError, "prefill"):
                    paired._context(experiment, MODEL)

    def test_complete_object_wrapper_is_accepted(self):
        experiment = SimpleNamespace(primary=Path("primary.json"), dataset=Path("dataset.json"),
                                     panel=(MODEL,), config={"wrapper_overrides": {}})
        context = self._context({"mode": "plain_completion"})
        with patch.object(paired, "_load_model_context", return_value=context):
            self.assertIs(paired._context(experiment, MODEL), context)

    def test_render_keeps_source_requests_and_refuses_truncation(self):
        context = self._context({})
        requests = [item["request"] for item in _rendered()]
        with patch.object(paired, "read_jsonl", return_value=iter(requests)), \
             patch.object(paired, "_load_renderer", return_value=(Mock(), Mock(encode=Mock(return_value=[1] * 1952)))), \
             patch.object(paired, "_render_prompt", return_value=("complete synthetic note prompt", 1952)):
            rendered = paired._render_requests(context)
        self.assertEqual([item["request"] for item in rendered], requests)
        with patch.object(paired, "read_jsonl", return_value=iter(requests)), \
             patch.object(paired, "_load_renderer", return_value=(Mock(), Mock(encode=Mock(return_value=[1] * 1953)))), \
             patch.object(paired, "_render_prompt", return_value=("complete synthetic note prompt", 1953)):
            with self.assertRaisesRegex(ValueError, "do not truncate"):
                paired._render_requests(context)

    def test_empty_duplicate_and_invalid_gold_requests_are_rejected(self):
        request = _rendered(1)[0]["request"]
        invalid = {**request, "expected": {**EXPECTED, "negation": "abstain"}}
        for requests in [[], [request, request], [invalid]]:
            with self.subTest(requests=requests), \
                 patch.object(paired, "read_jsonl", return_value=iter(requests)), \
                 patch.object(paired, "_load_renderer", return_value=(Mock(), Mock(encode=Mock(return_value=[1, 2, 3])))), \
                 patch.object(paired, "_render_prompt", return_value=("synthetic prompt", 3)):
                with self.assertRaises(ValueError):
                    paired._render_requests(self._context({}))

    def test_output_root_must_be_restricted_and_separate_from_legacy(self):
        with TemporaryDirectory() as temporary:
            restricted = Path(temporary) / "restricted"
            experiment = SimpleNamespace(dataset=restricted / "legacy/dataset/manifest.json")
            with patch.object(paired, "RESTRICTED_ROOT", restricted):
                self.assertEqual(paired._output_root(experiment, restricted / "paired"), restricted / "paired")
                for invalid in [Path(temporary) / "outside", restricted, restricted / "legacy", restricted / "legacy/subfolder"]:
                    with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                        paired._output_root(experiment, invalid)


if __name__ == "__main__":
    unittest.main()
