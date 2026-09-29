from __future__ import annotations

import json
from pathlib import Path
import unittest

from cpg2.slm_compact_contract import COMPACT_OUTPUT_KEYS, evaluate_contract_output
from cpg2.slm_e0 import STATE_REASON_CODES, parse_typed_e0_output


CRITERION = "current_fever"


def compact_payload(**overrides):
    payload = {
        "criterion_id": CRITERION,
        "state": "PRESENT_CURRENT",
        "confidence": 0.9,
        "abstain": False,
    }
    payload.update(overrides)
    return payload


def evaluate(payload, *, contract="compact", **kwargs):
    return evaluate_contract_output(
        json.dumps(payload),
        contract=contract,
        expected_criterion_id=CRITERION,
        **kwargs,
    )


class CompactContractTests(unittest.TestCase):
    def test_all_states_complete_only_declared_fields(self):
        for state, reason in STATE_REASON_CODES.items():
            for confidence in [0, 0.5, 1]:
                with self.subTest(state=state, confidence=confidence):
                    payload = compact_payload(state=state, confidence=confidence)
                    result = evaluate(payload)
                    self.assertIsNone(result["adapter_failure"])
                    assembled = json.loads(result["assembled_output"])
                    for key in COMPACT_OUTPUT_KEYS:
                        self.assertEqual(assembled[key], payload[key])
                    self.assertIsNone(assembled["value"])
                    self.assertIsNone(assembled["unit"])
                    self.assertEqual(assembled["evidence"], [])
                    self.assertEqual(assembled["reason_code"], reason)
                    self.assertTrue(result["parsed"]["logical_valid"])
                    self.assertEqual(result["parsed"]["state"], state)
                    self.assertEqual(result["state_only_state"], state)
                    self.assertEqual(
                        result["parsed"],
                        parse_typed_e0_output(
                            result["assembled_output"], expected_criterion_id=CRITERION
                        ).as_dict(),
                    )

    def test_valid_abstention_stays_abstention(self):
        result = evaluate(compact_payload(state="NOT_DOCUMENTED", confidence=0, abstain=True))
        self.assertTrue(result["parsed"]["logical_valid"])
        self.assertTrue(result["parsed"]["abstained"])
        self.assertIsNone(result["parsed"]["state"])
        self.assertEqual(result["state_only_state"], "NOT_DOCUMENTED")
        self.assertEqual(json.loads(result["assembled_output"])["reason_code"], "model_abstention")

    def test_contradictory_abstention_is_not_repaired(self):
        for state in STATE_REASON_CODES:
            for confidence in [0, 0.5, 1]:
                if state == "NOT_DOCUMENTED" and confidence == 0:
                    continue
                with self.subTest(state=state, confidence=confidence):
                    payload = compact_payload(state=state, confidence=confidence, abstain=True)
                    result = evaluate(payload)
                    self.assertIsNone(result["adapter_failure"])
                    self.assertEqual(result["parsed"]["failure"], "logical_failure")
                    self.assertTrue(result["parsed"]["schema_valid"])
                    self.assertTrue(result["parsed"]["abstained"])
                    assembled = json.loads(result["assembled_output"])
                    for key in COMPACT_OUTPUT_KEYS:
                        self.assertEqual(assembled[key], payload[key])

    def test_invalid_field_values_do_not_get_assembled(self):
        invalid_fields = {
            "criterion_id": ["wrong", None, 7, [], {}],
            "state": ["YES", "present_current", None, 7, [], {}],
            "confidence": [-0.1, 1.1, True, False, None, "0.9", [], {}, float("nan"), float("inf"), -float("inf")],
            "abstain": [None, 0, 1, "false", [], {}],
        }
        for key, values in invalid_fields.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    result = evaluate(compact_payload(**{key: value}))
                    self.assertEqual(result["adapter_failure"], "schema_failure")
                    self.assertEqual(result["parsed"]["failure"], "schema_failure")
                    self.assertTrue(result["parsed"]["parse_valid"])
                    self.assertIsNone(result["assembled_output"])
                    self.assertIsNone(result["parsed"]["state"])

    def test_missing_and_extra_keys_do_not_get_assembled(self):
        payloads = []
        for key in COMPACT_OUTPUT_KEYS:
            missing = compact_payload()
            del missing[key]
            payloads.append(missing)
        payloads.append(compact_payload(extra="ignored?"))
        payloads.append(compact_payload(value=None, unit=None, evidence=[], reason_code="explicit_present"))
        for payload in payloads:
            with self.subTest(keys=list(payload)):
                result = evaluate(payload)
                self.assertEqual(result["adapter_failure"], "schema_failure")
                self.assertFalse(result["parsed"]["schema_valid"])
                self.assertIsNone(result["assembled_output"])

    def test_malformed_json_is_never_extracted_or_repaired(self):
        valid = json.dumps(compact_payload())
        for raw in ["", "not json", valid[:-1], "```json\n" + valid + "\n```", "Here: " + valid, valid + valid, b"\xff"]:
            with self.subTest(raw=raw):
                result = evaluate_contract_output(raw, contract="compact", expected_criterion_id=CRITERION)
                self.assertEqual(result["adapter_failure"], "parse_failure")
                self.assertFalse(result["parsed"]["parse_valid"])
                self.assertIsNone(result["assembled_output"])
                self.assertIsNone(result["state_only_state"])

    def test_valid_json_nonobjects_are_schema_failures(self):
        for payload in [None, [], "text", 1, True]:
            with self.subTest(payload=payload):
                result = evaluate(payload)
                self.assertEqual(result["adapter_failure"], "schema_failure")
                self.assertTrue(result["parsed"]["parse_valid"])
                self.assertIsNone(result["state_only_state"])

    def test_runtime_failures_have_no_diagnostic_prediction(self):
        for contract in ["full", "compact"]:
            result = evaluate(compact_payload(), contract=contract, runtime_failed=True)
            self.assertEqual(result["parsed"]["failure"], "runtime_failure")
            self.assertIsNone(result["state_only_state"])
            result = evaluate_contract_output(None, contract=contract, expected_criterion_id=CRITERION)
            self.assertEqual(result["parsed"]["failure"], "runtime_failure")

    def test_state_only_ignores_nonstate_contract_failures(self):
        for contract in ["full", "compact"]:
            for payload in [
                {"criterion_id": CRITERION, "state": "PRESENT_CURRENT"},
                compact_payload(confidence="invalid", abstain="invalid"),
                compact_payload(extra="invalid"),
                compact_payload(abstain=True),
            ]:
                result = evaluate(payload, contract=contract)
                self.assertEqual(result["state_only_state"], "PRESENT_CURRENT")
                self.assertFalse(result["parsed"]["logical_valid"])
            for overrides in [{"criterion_id": "wrong"}, {"state": "YES"}, {"state": []}]:
                self.assertIsNone(evaluate(compact_payload(**overrides), contract=contract)["state_only_state"])

    def test_full_path_is_unchanged_validator(self):
        valid_full = json.loads(evaluate(compact_payload())["assembled_output"])
        cases = [valid_full, {**valid_full, "reason_code": "historical"}, {**valid_full, "confidence": True}, compact_payload(), None]
        for payload in cases:
            raw = json.dumps(payload)
            result = evaluate_contract_output(raw, contract="full", expected_criterion_id=CRITERION)
            self.assertEqual(result["assembled_output"], raw)
            self.assertIsNone(result["adapter_failure"])
            self.assertEqual(result["parsed"], parse_typed_e0_output(raw, expected_criterion_id=CRITERION).as_dict())

    def test_unknown_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_contract_output("{}", contract="unknown", expected_criterion_id=CRITERION)

    def test_prompt_clinical_instructions_and_retained_fields_are_unchanged(self):
        prompts = Path(__file__).resolve().parents[1] / "configs/slm_benchmark/prompts"
        full = json.loads((prompts / "typed_state.v1.json").read_text())
        compact = json.loads((prompts / "typed_state_compact.v1.json").read_text())
        clinical = full["system"].split(" Set value and unit to null", 1)[0]
        self.assertEqual(compact["system"].split(" Return only criterion_id", 1)[0], clinical)
        self.assertEqual(compact["user_template"], full["user_template"])
        self.assertEqual(set(compact["output_schema"]["required"]), COMPACT_OUTPUT_KEYS)
        self.assertFalse(compact["output_schema"]["additionalProperties"])
        for key in COMPACT_OUTPUT_KEYS:
            self.assertEqual(compact["output_schema"]["properties"][key], full["output_schema"]["properties"][key])


if __name__ == "__main__":
    unittest.main()
