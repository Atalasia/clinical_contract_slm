"""Deterministic completion of the compact typed-state response contract.

The adapter never changes a model-generated field.  It supplies only fields
whose values are fixed or redundant under the declared compact contract, then
uses the same full-record validator as the original experiment.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from .slm_e0 import (
    STATE_REASON_CODES,
    TYPED_STATES,
    TypedE0Prediction,
    parse_typed_e0_output,
)


COMPACT_OUTPUT_KEYS = frozenset({"criterion_id", "state", "confidence", "abstain"})


def _complete_json_object(raw_output: str | bytes | None) -> dict[str, Any] | None:
    """Use the original parser's complete-JSON semantics, without extraction."""

    try:
        if isinstance(raw_output, bytes):
            raw_output = raw_output.decode("utf-8")
        value = json.loads(raw_output)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _state_only(payload: dict[str, Any] | None, expected_criterion_id: str) -> str | None:
    if payload is None or payload.get("criterion_id") != expected_criterion_id:
        return None
    state = payload.get("state")
    return state if isinstance(state, str) and state in TYPED_STATES else None


def _adapter_failure(kind: str, *, parse_valid: bool) -> dict[str, Any]:
    # These are generated-contract failures, so no full record is assembled.
    # Match the unchanged full validator's failure representation explicitly.
    return TypedE0Prediction(
        state=None,
        parse_valid=parse_valid,
        schema_valid=False,
        logical_valid=False,
        abstained=False,
        failure=kind,
    ).as_dict()


def evaluate_contract_output(
    raw_output: str | bytes | None,
    *,
    contract: Literal["full", "compact"],
    expected_criterion_id: str,
    runtime_failed: bool = False,
) -> dict[str, Any]:
    """Evaluate raw model text, retaining generation failures and abstention.

    ``state_only_state`` is diagnostic: a matching ID and recognized state in a
    complete JSON object suffice, regardless of the other fields.  It is not a
    usable full-record prediction.  The raw output must be retained by callers
    separately from ``assembled_output``.

    Invalid compact fields do not yield an assembled record.  Schema-valid but
    logically contradictory abstention fields are passed through unchanged so
    the full validator rejects them.  In particular, the adapter never changes
    a state, confidence, ID, or abstention decision to make a record valid.
    """

    if contract not in {"full", "compact"}:
        raise ValueError("contract must be 'full' or 'compact'")
    payload = None if runtime_failed else _complete_json_object(raw_output)
    state_only_state = _state_only(payload, expected_criterion_id)
    if contract == "full":
        return {
            "parsed": parse_typed_e0_output(
                raw_output,
                expected_criterion_id=expected_criterion_id,
                runtime_failed=runtime_failed,
            ).as_dict(),
            "assembled_output": raw_output,
            "adapter_failure": None,
            "state_only_state": state_only_state,
        }

    failure: str | None = None
    parse_valid = False
    if runtime_failed or raw_output is None:
        failure = "runtime_failure"
    else:
        # An array, scalar, or null can be syntactically valid JSON even though
        # _complete_json_object intentionally does not expose it as a record.
        try:
            text = raw_output.decode("utf-8") if isinstance(raw_output, bytes) else raw_output
            json.loads(text)
            parse_valid = True
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            failure = "parse_failure"
        if failure is None:
            confidence = payload.get("confidence") if payload is not None else None
            compact_schema_valid = (
                payload is not None
                and set(payload) == COMPACT_OUTPUT_KEYS
                and isinstance(payload.get("criterion_id"), str)
                and payload["criterion_id"] == expected_criterion_id
                and isinstance(payload.get("state"), str)
                and payload["state"] in TYPED_STATES
                and isinstance(payload.get("abstain"), bool)
                and not isinstance(confidence, bool)
                and isinstance(confidence, (int, float))
                and 0 <= confidence <= 1
                and math.isfinite(confidence)
            )
            if not compact_schema_valid:
                failure = "schema_failure"
    if failure is not None:
        return {
            "parsed": _adapter_failure(failure, parse_valid=parse_valid),
            "assembled_output": None,
            "adapter_failure": failure,
            "state_only_state": state_only_state,
        }

    assert payload is not None
    assembled = {
        **payload,
        "value": None,
        "unit": None,
        "evidence": [],
        "reason_code": (
            "model_abstention" if payload["abstain"] else STATE_REASON_CODES[payload["state"]]
        ),
    }
    assembled_output = json.dumps(assembled, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "parsed": parse_typed_e0_output(
            assembled_output, expected_criterion_id=expected_criterion_id
        ).as_dict(),
        "assembled_output": assembled_output,
        "adapter_failure": None,
        "state_only_state": state_only_state,
    }
