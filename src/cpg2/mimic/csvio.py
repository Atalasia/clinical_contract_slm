from __future__ import annotations

import csv
import gzip
import hashlib
import hmac
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


def open_text(path: str | Path) -> TextIO:
    source = Path(path)
    if source.suffix.casefold() == ".gz":
        return gzip.open(source, mode="rt", encoding="utf-8", newline="")
    return source.open(mode="r", encoding="utf-8", newline="")


def read_header(path: str | Path) -> tuple[str, ...]:
    with open_text(path) as handle:
        reader = csv.reader(handle)
        try:
            return tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc


def iter_csv(path: str | Path) -> Iterator[tuple[int, dict[str, str]]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"CSV file has duplicate columns: {path}")
        for line_number, row in enumerate(reader, start=2):
            yield line_number, {key: value if value is not None else "" for key, value in row.items()}


def resolve_column(headers: Iterable[str], configured: str) -> str:
    exact = set(headers)
    if configured in exact:
        return configured
    matches = [value for value in exact if value.casefold() == configured.casefold()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"column {configured!r} is not present; available={sorted(exact)}")
    raise ValueError(f"column {configured!r} is ambiguous")


def resolve_columns(headers: Iterable[str], configured: Iterable[str]) -> dict[str, str]:
    header_tuple = tuple(headers)
    return {value: resolve_column(header_tuple, value) for value in configured}


def parse_fixed_timezone(value: str) -> timezone:
    if value in {"UTC", "Z", "+00:00"}:
        return timezone.utc
    if not isinstance(value, str) or len(value) != 6 or value[0] not in "+-" or value[3] != ":":
        raise ValueError("timestamp_timezone must be UTC, Z, or a fixed offset such as -05:00")
    try:
        hours, minutes = int(value[1:3]), int(value[4:6])
    except ValueError as exc:
        raise ValueError("timestamp_timezone has invalid digits") from exc
    if hours > 23 or minutes > 59:
        raise ValueError("timestamp_timezone offset is out of range")
    delta = timedelta(hours=hours, minutes=minutes)
    if value[0] == "-":
        delta = -delta
    return timezone(delta)


def parse_mimic_timestamp(value: str, *, timestamp_timezone: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("MIMIC timestamp is empty")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid MIMIC timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=parse_fixed_timezone(timestamp_timezone))
    return parsed


def isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.isoformat()


def load_hash_salt(path: str | Path) -> bytes:
    salt = Path(path).read_bytes().strip()
    if len(salt) < 16:
        raise ValueError("hash salt file must contain at least 16 non-whitespace bytes")
    return salt


def stable_local_key(salt: bytes, namespace: str, *values: object) -> str:
    message = "\x1f".join([namespace, *(str(value) for value in values)]).encode("utf-8")
    return hmac.new(salt, message, hashlib.sha256).hexdigest()


def config_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@contextmanager
def restricted_jsonl_writer(path: str | Path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yield handle
    finally:
        try:
            os.chmod(destination, 0o600)
        except OSError:
            pass


def write_restricted_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    finally:
        try:
            os.chmod(destination, 0o600)
        except OSError:
            pass


def write_jsonl_row(handle: TextIO, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            yield value


def resolve_path(value: str, *, base_dir: str | Path) -> Path:
    if not isinstance(value, str) or not value or value.startswith("__"):
        raise ValueError(f"path must be configured, got {value!r}")
    path = Path(value)
    return path if path.is_absolute() else Path(base_dir) / path
