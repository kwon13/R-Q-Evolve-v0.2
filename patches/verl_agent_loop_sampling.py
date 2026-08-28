"""Apply/check the per-call async-VERL sampling override required by v0.2.

The installed agent-loop worker otherwise ignores DataProto.meta_info values
for temperature, top-p, max tokens, logprobs, and allowed token IDs.  Domain
probabilities would then be neither binary nor calibrated.  The operation is
idempotent and preserves a `.orig` backup.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

MARKER = "# [rq-evolve-v0.2] per-call sampling overrides from meta_info v1"
LEGACY_MARKER = "# [rq-evolve] per-call sampling overrides from meta_info v2"
ANCHORS = (
    """        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
""",
    """        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
""",
)
INSERT = f"""
        {MARKER}
        for _rq_v02_key in (
            "temperature", "top_p", "top_k", "max_tokens",
            "logprobs", "allowed_token_ids",
        ):
            if _rq_v02_key in batch.meta_info:
                sampling_params[_rq_v02_key] = batch.meta_info[_rq_v02_key]
"""


def target_file() -> Path:
    import verl

    return Path(verl.__file__).parent / "experimental" / "agent_loop" / "agent_loop.py"


def is_applied(path: Path | None = None) -> bool:
    text = (path or target_file()).read_text(encoding="utf-8")
    return MARKER in text or LEGACY_MARKER in text


def apply() -> str:
    path = target_file()
    text = path.read_text(encoding="utf-8")
    if MARKER in text or LEGACY_MARKER in text:
        return f"already applied: {path}"
    anchor = next((value for value in ANCHORS if value in text), None)
    if anchor is None:
        raise RuntimeError(
            f"known patch anchor not found in {path}; derive a patch for this VERL version"
        )
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(anchor, anchor + INSERT, 1), encoding="utf-8")
    return f"patched {path}; backup: {backup}"


if __name__ == "__main__":
    if "--check" in sys.argv:
        print("applied" if is_applied() else "NOT APPLIED")
        raise SystemExit(0 if is_applied() else 1)
    print(apply())
