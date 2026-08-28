# R-Q-Evolve-v0.2

R-Q-Evolve-v0.2 is a clean sibling project for evolving **concrete competition-math problems**. It combines free-form two-parent crossover with a 7-domain × 5-problem-type MAP, pseudo-label validation, policy-relative \(R_Q\) scoring, and exact rollout replay.

The project was designed after inspecting the local `R-Zero` and `R-Q-Evolve` repositories, but it does not import either project at runtime. All production code, prompts, state, and launch paths live in this repository.

## Core contract

Each cycle has two distinct data paths:

1. Select 32 parent pairs from the accepted MAP and request two children per pair, giving 64 initial candidates. If too few survive, refill in blocks of 32 up to 128 candidates.
2. Strictly parse each candidate into one question and one proposed boxed answer.
3. Reject malformed, unsafe, exact-duplicate, near-duplicate, or parent-copy candidates.
4. Run exactly nine independent label rollouts. Invalid or boxless outputs remain in the fixed denominator. Accept only a unique pseudo-gold cluster with at least five votes and an independently verified match to the crossover model's proposed answer.
5. Assign exactly one high-confidence top-level mathematical domain and one deterministic problem type.
6. Append every valid and novel problem to the accepted catalog. Frontier membership controls training, not archival.
7. Score accepted incumbents with a complete group of eight current-policy rollouts and compute \(R_Q\).
8. Select training problems only with the previous iteration's score, then replay the newly measured current-policy payload exactly once. The fresh success rate is audit metadata, not a second eligibility gate: a complete current group remains eligible even when it is all-correct or all-wrong. A training update requires exactly 32 distinct problem groups from one policy version; there is no padding, resampling fallback, or partial update.

Fresh children first become eligible on the next cycle. Therefore, a clean discovery cycle can correctly archive problems while skipping the optimizer update.

See [docs/PIPELINE_CONTRACT.md](docs/PIPELINE_CONTRACT.md) for the invariants and [docs/REFERENCE_WIRING.md](docs/REFERENCE_WIRING.md) for design provenance.

## Installation

Use the same environment that provides the repository's compatible `verl`, Ray, PyTorch, vLLM, and CUDA stack:

```bash
cd /data1/yhoon113/R-Q-Evolve-v0.2
python -m pip install -e '.[train]'
```

For CPU-only unit tests:

```bash
python -m pip install -e '.[dev]'
pytest
```

The default model path in both production configs is `/data1/yhoon113/qwen3-4b-base`. Change `backend.model_path` and `verl_config.actor_rollout_ref.model.path` together if a server uses a different path.

## Preflight and launch

Four GPUs:

```bash
bash scripts/run_train_4gpu.sh --gpus 0,1,2,3
```

Eight GPUs, detached:

```bash
bash scripts/run_train_8gpu.sh --gpus 0,1,2,3,4,5,6,7 --detach
```

The launchers first apply the required async-VERL per-call sampling patch idempotently, then run the same CLI preflight used by manual operation. The patch preserves a `.orig` backup of the installed VERL source. Detached processes have one explicit PID file and one timestamped log; the scripts never use `pkill` or terminate an existing process.

The shipped configs deliberately cap Ray at 16 logical CPUs and a 16 GiB
object store. Leaving `ray_init.num_cpus` unset on large servers makes Ray
eagerly prestart one Python worker per host CPU before VERL loads the model.
Launchers also enable Python's fault handler, and the runtime emits unbuffered
startup markers around Ray, tokenizer, trainer, worker, and resident-fit
initialization so native failures retain a precise last completed boundary.

Equivalent manual commands are:

```bash
PYTHONPATH=src python patches/verl_agent_loop_sampling.py
PYTHONPATH=src python -m rq_evolve_v02 preflight --config configs/rq_evolve_v02_8gpu.yaml
PYTHONPATH=src python -m rq_evolve_v02 run --config configs/rq_evolve_v02_8gpu.yaml
```

Preflight fails closed when the installed VERL version has no known patch anchor or the marker is absent. The patch is required because crossover, label, score, and one-token domain requests use different per-call sampling parameters on the same resident async rollout worker.

## Persistent outputs

The pipeline owns an append-only accepted catalog and separate observations:

- `candidate_events.jsonl`: every acceptance or rejection decision and its reason;
- `accepted_problems.jsonl`: every valid, novel problem, not only frontier problems;
- `label_observations.jsonl`: terminal nine-rollout pseudo-label evidence for accepted and rejected candidates, including fixed-denominator clusters and the proposed-answer match verdict;
- `score_observations.jsonl`: policy-stamped \(R_Q\) measurements;
- `training_events.jsonl`: batch selection, replay, skip, and update events;
- `map_index.json`: rebuildable 35-cell projection over the accepted catalog;
- `pipeline_state.json`: committed iteration and checkpoint high-water marks;
- `iterations/`: cycle-local audit artifacts.

Rollout tensors used for an update are deliberately same-cycle, in-memory objects. After a crash they are remeasured rather than reconstructed from text or replayed off-policy. The run allows at most 320 cycles to obtain 256 successful optimizer updates. Skipped discovery or underfilled cycles consume the cycle budget but not the optimizer-step budget.

Every successful optimizer update is followed by a pipeline-orchestrated checkpoint while the resident fit loop is held at its post-weight-sync barrier. VERL's independent periodic checkpoint schedule is disabled with `save_freq: 0`; running both save paths would create a checkpoint before the pipeline state transaction is committed. The production configs retain the newest two actor/critic recovery points so per-update safety does not grow disk usage without bound.

## Resume policy

The supplied configs set `training.resume_mode: disable` for a clean first run. To resume deliberately, change only that flag to `auto`; it is treated as an operational flag rather than experiment identity. Keep the model path, archive directory, checkpoint directory, prompts, seeds, and every semantic config field unchanged. Preflight refuses an incompatible manifest or a silently reused output directory.
