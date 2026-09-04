"""Deterministic serialization, hashing, and numerical utilities."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize a JSON-compatible object with a stable byte representation."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(payload: object) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible object."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_json(payload: object, path: str | Path) -> None:
    """Write deterministic, human-readable JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> object:
    """Read JSON from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_float(value: object, *, name: str) -> float:
    """Validate and convert a finite numeric scalar."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def integer(value: object, *, name: str, minimum: int | None = None) -> int:
    """Validate an integer scalar without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def string(value: object, *, name: str) -> str:
    """Validate a nonempty string."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def as_object_dict(value: object, *, name: str) -> dict[str, Any]:
    """Validate a JSON object and return a typed dictionary."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return value
