"""Small deterministic helpers shared across pipeline stages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_SPACE_RE = re.compile(r"\s+")


def canonical_text(value: str) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: Any, length: int = 20) -> str:
    payload = stable_json(parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:length]
    return f"{prefix}_{digest}"


def value_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_seed(
    master_seed: int, namespace: str, iteration: int, index: int = 0
) -> int:
    raw = stable_json(
        [int(master_seed), namespace, int(iteration), int(index)]
    ).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0x7FFFFFFF
