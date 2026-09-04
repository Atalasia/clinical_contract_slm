from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .states import CriterionState, MergeOutcome


SCHEMA_VERSION = "1.0"
MISSING = object()


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def parse_datetime(value: str | datetime, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    patient_key: str
    encounter_key: str
    criterion_id: str
    source_type: str
    event_time: str
    available_time: str
    decision_cutoff: str
    source_table: str | None = None
    source_record_id: str | None = None
    note_id: str | None = None
    span_start: int | None = None
    span_end: int | None = None
    quoted_text_local_only: str | None = None
    raw_value: Any = None
    normalized_value: Any = None
    unit: str | None = None
    is_pre_cutoff: bool | None = None
    extraction_method: str = "deterministic"
    prompt_version: str | None = None
    model_version: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceRecord":
        if not isinstance(data, dict):
            raise ValueError("evidence record must be an object")
        required = {
            "evidence_id",
            "patient_key",
            "encounter_key",
            "criterion_id",
            "source_type",
            "event_time",
            "available_time",
            "decision_cutoff",
        }
        missing = required - data.keys()
        if missing:
            raise ValueError(f"evidence record missing fields: {sorted(missing)}")
        for key in required - {"event_time", "available_time", "decision_cutoff"}:
            _require_nonempty_string(data[key], key)
        if data["source_type"] not in {"structured", "note", "derived"}:
            raise ValueError("source_type must be structured, note, or derived")
        for key in ("event_time", "available_time", "decision_cutoff"):
            parse_datetime(data[key], key)
        start, end = data.get("span_start"), data.get("span_end")
        if (start is None) != (end is None):
            raise ValueError("span_start and span_end must both be supplied or both be null")
        if start is not None and (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError("evidence span must be non-negative with end > start")
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if public:
            result.pop("quoted_text_local_only", None)
        return result


@dataclass(frozen=True)
class CriterionAssessment:
    criterion_id: str
    state: CriterionState
    value: Any = None
    unit: str | None = None
    confidence: float | None = None
    evidence_ids: tuple[str, ...] = ()
    source_modalities: tuple[str, ...] = ()
    resolution_method: str = "unknown"
    is_pre_cutoff: bool | None = None
    abstained: bool = False
    parse_failure: bool = False
    inference_failure: bool = False
    conflict_details: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_string(self.criterion_id, "criterion_id")
        if not isinstance(self.state, CriterionState):
            object.__setattr__(self, "state", CriterionState(self.state))
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be numeric and in [0, 1]")
        expected_flags = {
            CriterionState.MODEL_ABSTENTION: self.abstained,
            CriterionState.PARSE_FAILURE: self.parse_failure,
            CriterionState.INFERENCE_FAILURE: self.inference_failure,
        }
        for state, flag in expected_flags.items():
            if self.state == state and not flag:
                raise ValueError(f"{state.value} requires its matching failure/abstention flag")
        if self.abstained and self.state != CriterionState.MODEL_ABSTENTION:
            raise ValueError("abstained=true requires MODEL_ABSTENTION")
        if self.parse_failure and self.state != CriterionState.PARSE_FAILURE:
            raise ValueError("parse_failure=true requires PARSE_FAILURE")
        if self.inference_failure and self.state != CriterionState.INFERENCE_FAILURE:
            raise ValueError("inference_failure=true requires INFERENCE_FAILURE")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionAssessment":
        if not isinstance(data, dict):
            raise ValueError("criterion assessment must be an object")
        if "criterion_id" not in data or "state" not in data:
            raise ValueError("criterion assessment requires criterion_id and state")
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown assessment fields: {sorted(unknown)}")
        normalized = dict(data)
        normalized["state"] = CriterionState(data["state"])
        for field_name in ("evidence_ids", "source_modalities", "conflict_details"):
            value = normalized.get(field_name, ())
            if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) for x in value):
                raise ValueError(f"{field_name} must be a list of strings")
            normalized[field_name] = tuple(value)
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        for key in ("evidence_ids", "source_modalities", "conflict_details"):
            result[key] = list(result[key])
        return result


@dataclass(frozen=True)
class MergeResult:
    assessment: CriterionAssessment
    outcome: MergeOutcome
    source_assessments: dict[str, CriterionAssessment] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment": self.assessment.to_dict(),
            "outcome": self.outcome.value,
            "source_assessments": {
                key: value.to_dict() for key, value in sorted(self.source_assessments.items())
            },
        }
