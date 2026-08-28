"""Persistent main-thread math_verify worker; the parent can hard-kill it."""

from __future__ import annotations

import json
import sys

from verifier import canonical_boolean, normalize_verifier, parse_finite_set


def _parse(value: str):
    from math_verify import parse

    return parse(r"\boxed{" + str(value) + "}")


def _equal(left: str, right: str) -> bool:
    from math_verify import verify

    return bool(verify(_parse(right), _parse(left)))


def _set_equal(pred: str, gold: str) -> bool:
    left, right = parse_finite_set(pred), parse_finite_set(gold)
    if left is None or right is None or len(left) != len(right):
        return False
    used: set[int] = set()
    for item in left:
        matches = [
            i
            for i, target in enumerate(right)
            if i not in used and _equal(item, target)
        ]
        if not matches:
            return False
        used.add(matches[0])
    return True


def grade(pred: str, gold: str, verifier: dict | None) -> bool:
    spec = normalize_verifier(verifier, answer=gold)
    if spec["mode"] == "expression":
        return _equal(pred, gold)
    if spec["mode"] == "boolean":
        return (
            canonical_boolean(pred) == canonical_boolean(gold)
            and canonical_boolean(gold) is not None
        )
    if spec["mode"] == "set":
        return _set_equal(pred, gold)
    return False


def main() -> None:
    # Warm imports before acknowledging readiness.
    from math_verify import parse  # noqa: F401

    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            result = grade(
                str(request["pred"]), str(request["gold"]), request.get("verifier")
            )
            print(json.dumps({"ok": True, "match": bool(result)}), flush=True)
        except Exception as exc:
            print(
                json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}),
                flush=True,
            )


if __name__ == "__main__":
    main()
