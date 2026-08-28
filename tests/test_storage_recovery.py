from __future__ import annotations

import json
from pathlib import Path

import pytest

from rq_evolve_v02.storage import (
    PipelineState,
    RunStore,
    append_jsonl,
    read_jsonl,
    repair_truncated_jsonl_tail,
)


def test_truncated_final_row_is_discarded_before_next_append(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"row":1}\n{"row":')
    assert list(read_jsonl(path)) == [{"row": 1}]

    append_jsonl(path, {"row": 2}, fsync=False)
    assert list(read_jsonl(path)) == [{"row": 1}, {"row": 2}]
    assert path.read_bytes().endswith(b"\n")
    assert b'{"row":\n' not in path.read_bytes()


def test_complete_unterminated_row_gets_newline_not_deleted(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_bytes(b'{"score":0.5}')
    assert repair_truncated_jsonl_tail(path, fsync=False)
    assert path.read_bytes() == b'{"score":0.5}\n'
    assert list(read_jsonl(path)) == [{"score": 0.5}]


def test_partial_utf8_tail_is_recoverable(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    valid = json.dumps({"text": "정상"}, ensure_ascii=False).encode("utf-8") + b"\n"
    path.write_bytes(valid + b'{"text":"\xed')
    assert list(read_jsonl(path)) == [{"text": "정상"}]
    assert repair_truncated_jsonl_tail(path, fsync=False)
    assert path.read_bytes() == valid


def test_newline_terminated_or_interior_corruption_is_never_hidden(
    tmp_path: Path,
) -> None:
    final = tmp_path / "final.jsonl"
    final.write_bytes(b'{"ok":1}\nnot-json\n')
    with pytest.raises(json.JSONDecodeError):
        list(read_jsonl(final))

    interior = tmp_path / "interior.jsonl"
    interior.write_bytes(b'{"ok":1}\nnot-json\n{"ok":2}\n')
    with pytest.raises(json.JSONDecodeError):
        list(read_jsonl(interior))


def test_runstore_resume_eagerly_repairs_every_append_only_log(tmp_path: Path) -> None:
    manifest = {"schema_version": "concrete-problem-map-v1", "run_uuid": "repair-run"}
    first = RunStore(tmp_path, fsync_jsonl=False)
    first.initialize(manifest, resume=False)
    paths = (
        first.events_path,
        first.accepted_path,
        first.labels_path,
        first.scores_path,
        first.training_path,
    )
    for index, path in enumerate(paths):
        path.write_bytes(f'{{"committed":{index}}}\n{{"partial":'.encode())

    resumed = RunStore(tmp_path, fsync_jsonl=False)
    resumed.initialize(manifest, resume=True)
    for index, path in enumerate(paths):
        assert list(read_jsonl(path)) == [{"committed": index}]
        assert path.read_bytes().endswith(b"\n")
    assert resumed.load_state() == PipelineState()


def test_checkpoint_event_prefix_allows_suffix_but_rejects_rewrite(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "prefix", fsync_jsonl=False)
    manifest = {"schema_version": "concrete-problem-map-v1", "run_uuid": "prefix"}
    store.initialize(manifest, resume=False)
    append_jsonl(store.events_path, {"event_id": "committed"}, fsync=False)
    size, digest = store.event_position()
    state = PipelineState(
        iteration=1,
        policy_version=1,
        global_step=1,
        checkpoint_step=1,
        checkpoint_event_offset=size,
        checkpoint_event_hash=digest,
    )
    store.save_state(state)
    append_jsonl(store.events_path, {"event_id": "later"}, fsync=False)
    store.verify_checkpoint_event_prefix(store.load_state())

    data = store.events_path.read_bytes()
    store.events_path.write_bytes(b"X" + data[1:])
    with pytest.raises(ValueError, match="prefix hash differs"):
        store.verify_checkpoint_event_prefix(store.load_state())


def test_checkpoint_prefix_detects_truncation_below_offset(tmp_path: Path) -> None:
    store = RunStore(tmp_path, fsync_jsonl=False)
    store.initialize(
        {"schema_version": "concrete-problem-map-v1", "run_uuid": "short-run"},
        resume=False,
    )
    store.events_path.write_bytes(b'{"event_id":"one"}\n')
    offset, digest = store.event_position()
    state = PipelineState(
        iteration=1,
        phase="ready",
        policy_version=1,
        global_step=1,
        checkpoint_step=1,
        checkpoint_event_offset=offset,
        checkpoint_event_hash=digest,
    )
    store.save_state(state)
    store.events_path.write_bytes(b"")
    with pytest.raises(ValueError, match="shorter than the checkpoint"):
        store.load_state()


def test_training_events_are_idempotent_by_event_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "training-dedupe", fsync_jsonl=False)
    store.initialize(
        {"schema_version": "concrete-problem-map-v1", "run_uuid": "dedupe"},
        resume=False,
    )
    row = {"event_id": "abort-1", "iteration": 2, "status": "aborted"}
    assert store.append_training_event(row)
    assert not store.append_training_event(dict(row))
    assert list(read_jsonl(store.training_path)) == [row]
