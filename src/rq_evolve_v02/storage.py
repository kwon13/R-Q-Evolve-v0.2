"""Append-only run storage with atomic indexes and resumable cycle artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import CandidateEvent, LabelEvidence, ProblemRecord, ScoreEvidence
from .utils import stable_json


SCHEMA_VERSION = "concrete-problem-map-v1"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def append_jsonl(path: Path, value: Any, *, fsync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    repair_truncated_jsonl_tail(path, fsync=fsync)
    encoded = stable_json(value) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def repair_truncated_jsonl_tail(path: Path, *, fsync: bool = True) -> bool:
    """Repair only an interrupted final row before the next append.

    A malformed interior row is never hidden. A complete final JSON object
    merely missing its newline receives one; an incomplete final object is
    truncated back to the preceding newline.
    """

    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r+b") as handle:
        data = handle.read()
        if data.endswith(b"\n"):
            return False
        boundary = data.rfind(b"\n") + 1
        tail = data[boundary:]
        try:
            decoded = json.loads(tail.decode("utf-8"))
            complete = isinstance(decoded, dict)
        except (UnicodeDecodeError, json.JSONDecodeError):
            complete = False
        if complete:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
        else:
            handle.truncate(boundary)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    return True


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    data = path.read_bytes()
    lines = data.splitlines()
    terminated = data.endswith(b"\n")
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8")
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # A process can die after writing only part of the final line.  A
            # newline-terminated malformed row is committed corruption, even
            # when it happens to be last, and must never be hidden.
            if index == len(lines) - 1 and not terminated:
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {index + 1} in {path} is not an object")
        yield value


@dataclass(slots=True)
class PipelineState:
    schema_version: str = SCHEMA_VERSION
    iteration: int = 0
    phase: str = "ready"
    active_cycle_id: str | None = None
    active_training_batch_id: str | None = None
    policy_version: int = 0
    global_step: int = 0
    checkpoint_step: int = 0
    checkpoint_event_offset: int = 0
    checkpoint_event_hash: str = ""

    def validate(self) -> None:
        allowed_phases = {
            "ready",
            "policy_frozen",
            "scoring",
            "discovery",
            "batch_ready",
            "update_applied",
            "checkpointed",
        }
        if self.phase not in allowed_phases:
            raise ValueError(f"unknown pipeline phase: {self.phase!r}")
        counters = {
            "iteration": self.iteration,
            "policy_version": self.policy_version,
            "global_step": self.global_step,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_event_offset": self.checkpoint_event_offset,
        }
        if any(
            isinstance(value, bool) or int(value) < 0 for value in counters.values()
        ):
            raise ValueError(
                f"pipeline counters must be nonnegative integers: {counters}"
            )
        if self.checkpoint_step > self.global_step:
            raise ValueError("checkpoint_step cannot exceed global_step")
        if self.phase != "update_applied" and self.checkpoint_step != self.global_step:
            raise ValueError(
                "only update_applied may be ahead of the durable checkpoint high-water mark"
            )
        if self.phase == "ready":
            if (
                self.active_cycle_id is not None
                or self.active_training_batch_id is not None
            ):
                raise ValueError("ready state cannot retain active cycle or batch IDs")
        elif not self.active_cycle_id:
            raise ValueError(f"phase {self.phase!r} requires active_cycle_id")
        if self.phase in {"batch_ready", "update_applied", "checkpointed"}:
            if not self.active_training_batch_id:
                raise ValueError(
                    f"phase {self.phase!r} requires active_training_batch_id"
                )
        elif self.active_training_batch_id is not None:
            raise ValueError(
                f"phase {self.phase!r} cannot retain active_training_batch_id"
            )
        if self.checkpoint_step == 0:
            if self.checkpoint_event_offset != 0 or self.checkpoint_event_hash:
                raise ValueError(
                    "zero checkpoint_step cannot carry an event high-water mark"
                )
        elif not self.checkpoint_event_hash:
            raise ValueError("durable checkpoint state requires checkpoint_event_hash")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PipelineState":
        return cls(**value)


class RunStore:
    """Single-writer persistence boundary for one clean run."""

    def __init__(
        self,
        root: str | Path,
        *,
        fsync_jsonl: bool = True,
        store_rollout_text: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.fsync_jsonl = bool(fsync_jsonl)
        self.store_rollout_text = bool(store_rollout_text)
        self.events_path = self.root / "candidate_events.jsonl"
        self.accepted_path = self.root / "accepted_problems.jsonl"
        self.labels_path = self.root / "label_observations.jsonl"
        self.scores_path = self.root / "score_observations.jsonl"
        self.training_path = self.root / "training_events.jsonl"
        self.map_path = self.root / "map_index.json"
        self.state_path = self.root / "pipeline_state.json"
        self.manifest_path = self.root / "run_manifest.json"
        self.iterations_dir = self.root / "iterations"
        self._event_ids: set[str] | None = None
        self._problem_ids: set[str] | None = None
        self._score_ids: set[str] | None = None
        self._label_ids: set[str] | None = None
        self._training_ids: set[str] | None = None

    def repair_truncated_logs(self) -> list[str]:
        """Repair interrupted final rows in every append-only run log.

        This is run eagerly on resume, not only on the next append, so a log
        that receives no further rows cannot retain a permanently ignored
        garbage tail. Interior or newline-terminated corruption still fails
        later reads and is never rewritten.
        """

        repaired: list[str] = []
        for path in (
            self.events_path,
            self.accepted_path,
            self.labels_path,
            self.scores_path,
            self.training_path,
        ):
            if repair_truncated_jsonl_tail(path, fsync=self.fsync_jsonl):
                repaired.append(path.name)
        return repaired

    def initialize(self, manifest: dict[str, Any], *, resume: bool) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if previous != manifest:
                raise ValueError(
                    "run manifest differs from the existing output directory; "
                    "use a new directory or resume with the identical config"
                )
            if not resume:
                raise ValueError(
                    f"refusing to reuse non-empty run directory: {self.root}"
                )
            self.repair_truncated_logs()
        else:
            occupied = list(self.root.iterdir())
            if occupied:
                raise ValueError(
                    f"output directory is non-empty without a manifest: {self.root}"
                )
            atomic_write_json(self.manifest_path, manifest)
            self.save_state(PipelineState())

    def load_state(self) -> PipelineState:
        if not self.state_path.exists():
            return PipelineState()
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        state = PipelineState.from_dict(value)
        if state.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported state schema {state.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        state.validate()
        self.verify_checkpoint_event_prefix(state)
        return state

    def save_state(self, state: PipelineState) -> None:
        state.validate()
        atomic_write_json(self.state_path, state.to_dict())

    def append_event(self, event: CandidateEvent) -> bool:
        if self._event_ids is None:
            self._event_ids = {
                str(row["event_id"]) for row in read_jsonl(self.events_path)
            }
        if event.event_id in self._event_ids:
            return False
        append_jsonl(self.events_path, event.to_dict(), fsync=self.fsync_jsonl)
        self._event_ids.add(event.event_id)
        return True

    def append_problem(self, record: ProblemRecord) -> bool:
        if self._problem_ids is None:
            self._problem_ids = {
                str(row["problem_id"]) for row in read_jsonl(self.accepted_path)
            }
        if record.problem_id in self._problem_ids:
            return False
        append_jsonl(
            self.accepted_path,
            record.to_dict(include_rollouts=False),
            fsync=self.fsync_jsonl,
        )
        self._problem_ids.add(record.problem_id)
        return True

    def append_score(self, problem_id: str, score: ScoreEvidence) -> bool:
        score_id = (
            score.observation_id
            or f"{problem_id}:{score.iteration}:{score.policy_version}"
        )
        score.observation_id = score_id
        if self._score_ids is None:
            self._score_ids = {
                str(row["score_id"]) for row in read_jsonl(self.scores_path)
            }
        if score_id in self._score_ids:
            return False
        append_jsonl(
            self.scores_path,
            {
                "score_id": score_id,
                "problem_id": problem_id,
                "score": score.to_dict(include_rollouts=self.store_rollout_text),
            },
            fsync=self.fsync_jsonl,
        )
        self._score_ids.add(score_id)
        return True

    def append_label_observation(
        self,
        *,
        observation_id: str,
        candidate_id: str,
        iteration: int,
        evidence: LabelEvidence,
    ) -> bool:
        if self._label_ids is None:
            self._label_ids = {
                str(row["label_observation_id"]) for row in read_jsonl(self.labels_path)
            }
        if observation_id in self._label_ids:
            return False
        payload = evidence.to_dict()
        if not self.store_rollout_text:
            for rollout in payload.get("rollouts", []):
                rollout["response"] = ""
        append_jsonl(
            self.labels_path,
            {
                "label_observation_id": observation_id,
                "candidate_id": candidate_id,
                "iteration": int(iteration),
                "evidence": payload,
            },
            fsync=self.fsync_jsonl,
        )
        self._label_ids.add(observation_id)
        return True

    def append_training_event(self, value: dict[str, Any]) -> bool:
        event_id = str(value.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("training event requires a non-empty event_id")
        if self._training_ids is None:
            self._training_ids = {
                str(row["event_id"]) for row in read_jsonl(self.training_path)
            }
        if event_id in self._training_ids:
            return False
        append_jsonl(self.training_path, value, fsync=self.fsync_jsonl)
        self._training_ids.add(event_id)
        return True

    def load_problems(self) -> list[ProblemRecord]:
        return [ProblemRecord.from_dict(row) for row in read_jsonl(self.accepted_path)]

    def load_scores(self) -> Iterator[tuple[str, ScoreEvidence]]:
        for row in read_jsonl(self.scores_path):
            yield str(row["problem_id"]), ScoreEvidence.from_dict(row["score"])

    def cycle_dir(self, iteration: int) -> Path:
        return self.iterations_dir / f"iter_{int(iteration):06d}"

    def write_cycle_artifact(self, iteration: int, name: str, value: Any) -> Path:
        path = self.cycle_dir(iteration) / f"{name}.json"
        atomic_write_json(path, value)
        return path

    def read_cycle_artifact(self, iteration: int, name: str) -> Any | None:
        path = self.cycle_dir(iteration) / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write_map_index(self, value: dict[str, Any]) -> None:
        atomic_write_json(self.map_path, value)

    def event_position(self) -> tuple[int, str]:
        if not self.events_path.exists():
            return 0, ""
        data = self.events_path.read_bytes()
        import hashlib

        return len(data), hashlib.sha256(data).hexdigest()

    def verify_checkpoint_event_prefix(self, state: PipelineState) -> None:
        """Verify the candidate-event prefix named by a durable checkpoint.

        Rows appended after the checkpoint are allowed. Missing, shortened, or
        rewritten bytes at or before its high-water mark are corruption and
        make automatic resume unsafe.
        """

        if state.checkpoint_step == 0:
            return
        import hashlib

        data = self.events_path.read_bytes() if self.events_path.exists() else b""
        offset = int(state.checkpoint_event_offset)
        if offset > len(data):
            raise ValueError(
                "candidate event log is shorter than the checkpoint high-water mark"
            )
        actual = hashlib.sha256(data[:offset]).hexdigest()
        if actual != state.checkpoint_event_hash:
            raise ValueError(
                "candidate event prefix hash differs from the durable checkpoint"
            )
        if offset and data[offset - 1 : offset] != b"\n":
            raise ValueError(
                "checkpoint event offset does not end at a JSONL row boundary"
            )
