"""One consistent, hard-time-limited typed answer comparator."""

from __future__ import annotations

import json
from pathlib import Path
import selectors
import subprocess
import sys
import threading
from typing import Any

from .verifier import normalize_verifier


class GraderClient:
    def __init__(self, *, timeout_s: float = 8.0) -> None:
        self.timeout_s = float(timeout_s)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _start(self) -> subprocess.Popen[str]:
        worker = Path(__file__).with_name("_grader_worker.py")
        process = subprocess.Popen(
            [sys.executable, "-u", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # The worker protocol is stdout-only.  Leaving stderr as an unread
            # pipe can eventually fill during a long run and deadlock grading.
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("grader worker has no stdout")
        ready = process.stdout.readline()
        if not ready or not json.loads(ready).get("ready"):
            process.kill()
            raise RuntimeError("grader worker failed to initialize")
        self._process = process
        return process

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        with self._lock:
            self._stop()

    def grade(self, pred: str, gold: str, verifier: dict[str, Any] | None) -> bool:
        try:
            spec = normalize_verifier(verifier, answer=str(gold))
        except (TypeError, ValueError):
            return False
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                process = self._start()
            assert process.stdin is not None and process.stdout is not None
            request = json.dumps(
                {"pred": str(pred), "gold": str(gold), "verifier": spec}
            )
            try:
                process.stdin.write(request + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                self._stop()
                return False
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            events = selector.select(self.timeout_s)
            selector.close()
            if not events:
                self._stop()
                return False
            line = process.stdout.readline()
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self._stop()
                return False
            return bool(response.get("ok") and response.get("match"))

    def equivalent(self, left: str, right: str, verifier: dict[str, Any]) -> bool:
        if str(left).strip() == str(right).strip():
            return True
        # Equivalence classes used by pseudo-labeling must be symmetric.  The
        # ordinary prediction-vs-gold reward remains directional.
        return self.grade(left, right, verifier) and self.grade(right, left, verifier)


_DEFAULT: GraderClient | None = None


def answers_match(pred: str, gold: str, verifier: dict[str, Any] | None = None) -> bool:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GraderClient()
    return _DEFAULT.grade(pred, gold, verifier)
