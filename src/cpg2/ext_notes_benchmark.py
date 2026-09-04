from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


OUTPUT_KEYS = frozenset({"detection", "encounter", "negation"})
DETECTION_LABELS = frozenset({"yes", "no", "abstain"})
ENCOUNTER_LABELS = frozenset({"yes", "no", "not_applicable", "abstain"})
NEGATION_LABELS = frozenset({"yes", "no", "unsure", "not_applicable", "abstain"})


@dataclass(frozen=True)
class ExtNotesPrediction:
    detection: str | None
    encounter: str | None
    negation: str | None
    schema_valid: bool
    logical_valid: bool
    failure: str | None = None

    @property
    def abstained(self) -> bool:
        return "abstain" in {self.detection, self.encounter, self.negation}

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["abstained"] = self.abstained
        return result


def _failure(kind: str, *, schema_valid: bool = False) -> ExtNotesPrediction:
    return ExtNotesPrediction(
        detection=None,
        encounter=None,
        negation=None,
        schema_valid=schema_valid,
        logical_valid=False,
        failure=kind,
    )


def validate_ext_notes_labels(
    payload: Mapping[str, Any],
    *,
    allow_abstention: bool,
) -> str | None:
    """Return a validation error, keeping dataset ``unsure`` distinct from abstention."""

    if set(payload) != OUTPUT_KEYS:
        return "output keys do not match the frozen E3 contract"
    detection = payload.get("detection")
    encounter = payload.get("encounter")
    negation = payload.get("negation")
    if not isinstance(detection, str) or detection not in DETECTION_LABELS:
        return "invalid detection label"
    if not isinstance(encounter, str) or encounter not in ENCOUNTER_LABELS:
        return "invalid encounter label"
    if not isinstance(negation, str) or negation not in NEGATION_LABELS:
        return "invalid negation label"
    if not allow_abstention and "abstain" in {detection, encounter, negation}:
        return "gold or expected labels cannot use model abstention"
    if detection == "no" and (
        encounter != "not_applicable" or negation != "not_applicable"
    ):
        return "detection=no requires not_applicable downstream labels"
    if detection == "yes" and encounter not in {"yes", "no", "abstain"}:
        return "detection=yes requires an encounter classification or abstention"
    if detection == "yes" and negation not in {"yes", "no", "unsure", "abstain"}:
        return "detection=yes requires a negation classification or abstention"
    if detection == "abstain" and (
        encounter != "abstain" or negation != "abstain"
    ):
        return "detection abstention requires downstream abstention"
    return None


def parse_ext_notes_output(
    raw_output: str | bytes | None,
    *,
    runtime_failed: bool = False,
) -> ExtNotesPrediction:
    """Strictly parse the candidate-conditioned E3 native-label output contract."""

    if runtime_failed or raw_output is None:
        return _failure("runtime_failure")
    try:
        if isinstance(raw_output, bytes):
            raw_output = raw_output.decode("utf-8")
        payload = json.loads(raw_output)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return _failure("parse_failure")
    if not isinstance(payload, dict):
        return _failure("schema_failure")
    if set(payload) != OUTPUT_KEYS:
        return _failure("schema_failure")
    detection = payload.get("detection")
    encounter = payload.get("encounter")
    negation = payload.get("negation")
    if (
        not isinstance(detection, str)
        or detection not in DETECTION_LABELS
        or not isinstance(encounter, str)
        or encounter not in ENCOUNTER_LABELS
        or not isinstance(negation, str)
        or negation not in NEGATION_LABELS
    ):
        return _failure("schema_failure")
    error = validate_ext_notes_labels(payload, allow_abstention=True)
    prediction = ExtNotesPrediction(
        detection=payload["detection"],
        encounter=payload["encounter"],
        negation=payload["negation"],
        schema_valid=True,
        logical_valid=error is None,
        failure=None if error is None else "logical_failure",
    )
    return prediction
