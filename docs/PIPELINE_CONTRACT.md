# Pipeline contract

This document defines correctness conditions. If a stage cannot prove its output satisfies the next stage's input contract, it must reject, abstain, rescore, or skip an update. It must not guess or silently relax the contract.

## Axes and archive semantics

The MAP is the Cartesian product of seven top-level mathematical domains and five computational output types. Every accepted problem occupies exactly one cell.

- Domain answers “which mathematical area is primary?” and is assigned independently from generation.
- Problem type answers “what output contract is requested?” and is one of `decision`, `search`, `counting`, `optimization`, or `function`.
- A problem with no unique high-confidence domain or no unambiguous supported output contract is rejected.
- The accepted catalog contains **every** problem that passes validity and novelty. A representative per cell is only an index view, never a deletion policy.
- Rejections remain auditable in the candidate-event log and never enter the accepted catalog.

## Cycle state machine

For policy \(\theta_t\), one committed cycle is:

1. Load the committed catalog, MAP projection, prior score observations, and policy identity.
2. Select a bounded, cell-balanced incumbent set for current-policy remeasurement.
3. Generate one complete score group of size \(G=8\) per incumbent, grade it, compute \(R_Q\), and retain the exact opaque rollout payload in memory.
4. Form the training set using only eligibility and priority information available before cycle \(t\). The current measurement supplies the on-policy payload but does not select itself. Freeze exactly 32 distinct, complete groups from the same policy version or record a skipped update.
5. Draw exactly 32 deterministic parent pairs and generate two children from each distinct prompt, giving 64 initial candidates.
6. Parse, validate, test novelty, pseudo-label, classify, and append all surviving candidates. Generate refill blocks of 32 only when required, never exceeding 128 candidates in the cycle.
7. Persist child score observations for MAP analysis and future selection. A child created or first scored in cycle \(t\) is not eligible for cycle \(t\)'s training batch.
8. If and only if the frozen replay batch has exactly 32 complete groups, perform one optimizer update from those payloads. Otherwise append a structured skip event and leave weights unchanged.
9. After every successful optimizer update, invoke the pipeline's external checkpoint transaction at the post-weight-sync barrier and commit its state high-water marks. VERL's independent periodic saver remains disabled.

Discovery and training are therefore concurrent pipeline stages with a one-cycle eligibility delay, not score-and-immediately-train on the same newly selected noise realization.

## Parent pairing and crossover

- Parent selection first samples occupied cells, then problems within cells, so large cells do not receive probability merely because they contain more rows.
- A pair should have disjoint lineage roots when the catalog permits it.
- A parent cannot be paired with itself.
- Pair IDs, prompt seeds, and child indices are deterministic under `run.master_seed` and iteration.
- The initial budget is invariant: `32 pairs × 2 children = 64 candidates`.
- The crossover model receives no target domain, problem type, destination cell, or requested mutation direction.
- Each accepted response contains exactly one question and exactly one proposed boxed answer. Extra prose or multiple envelopes is a parse failure.

## Validity, pseudo-labeling, and novelty

Labeling uses a separate rollout request from scoring. Its payload is never eligible for policy training.

For each candidate:

1. Run exactly nine solver rollouts under the label policy identity recorded for the cycle.
2. Extract at most one final boxed answer from each response.
3. Keep the denominator equal to nine. A completed response with a missing, invalid, or ungradeable boxed answer contributes no vote but still occupies a denominator slot. Transport timeout, worker loss, truncation at the token cap, or another infrastructure failure rejects or reruns the whole label group; it never creates a smaller denominator.
4. Build equivalence clusters using symmetric, hard-timeout grading. A non-transitive equivalence graph, tied largest clusters, or grader failure causes abstention.
5. Require a unique cluster of at least five answers, so `agreement >= 5/9`.
6. Require the crossover model's proposed answer to grade equivalent to the pseudo-gold representative.
7. Reject proof-only, damaged, underspecified, unsafe, or unsupported-output questions.

After infrastructure retries are exhausted or a semantic verdict is reached, persist exactly one terminal record in `label_observations.jsonl`. The row is keyed by `label_observation_id`, candidate, and iteration and contains the nine-rollout evidence, cluster sizes, agreement, proposed-answer match, acceptance flag, and reason. Both accepted and rejected terminal observations are retained. With `archive.store_rollout_text: false`, response bodies are blanked but the verdict metadata remains; the supplied production configs keep them.

Novelty is checked against both parents, the accepted catalog, and earlier survivors in the same cycle. It combines normalized exact equality, template/skeleton similarity, shingle or SimHash candidate lookup, and a final high-similarity comparison. Accepted candidates are inserted into the live novelty index immediately, preventing siblings from bypassing the check together.

## Domain and problem-type assignment

Domain labeling evaluates all seven domain hypotheses independently and accepts exactly one only when both configured confidence conditions hold:

- top probability at least `domain.min_probability`;
- every other arm below `domain.min_probability`, so there is exactly one high-confidence YES arm;
- top-versus-runner-up logit margin at least `domain.min_logit_margin`.

The labeler identity, policy version, prompt/rules hash, seven probabilities, selected label, and thresholds belong in the audit evidence. A generated self-declared domain is ignored.

Problem type is derived from the visible output request and cross-checked against the verifier mode:

- `decision` → Boolean (`Yes`/`No`);
- `search` → a complete finite set;
- `counting`, `optimization`, and `function` → a scalar/expression result.

Problem type does not direct crossover. It describes the accepted child after generation.

## Score observation

A score observation is valid only for one exact tuple:

```text
(problem_id, prompt_hash, pseudo_gold_hash, verifier_hash,
 policy_version, checkpoint_identity, sampling_contract, G)
```

All \(G=8\) rows must be present. A boxless completed response is a wrong answer, not a dropped row. Timeout, worker loss, stale-policy output, or incomplete payload invalidates the complete group and requires a rescore.

With \(c\) correct responses:

\[
\hat{s}=\frac{c}{G},\qquad
L=\frac{G}{G-1}\hat{s}(1-\hat{s}),\qquad
R_Q=L\,U.
\]

\(U\) is the mean normalized token entropy computed from the same rollout rows. If required entropy metadata is absent or non-finite, the observation is invalid. Scores are policy-relative observations, not permanent properties of a problem and not values to average across checkpoints.

## Frontier selection and exact replay

Selection is deliberately lagged by one cycle:

- The score observation from exactly cycle \(t-1\) establishes eligibility and priority.
- Cycle \(t\) remeasures that incumbent under \(\theta_t\).
- The cycle-\(t\) group must be complete, valid, and sampled under \(\theta_t\), but its fresh `s_hat` does not gate its own inclusion. It may be all-correct or all-wrong; such a group has zero GRPO advantage, while using it preserves selection independence.
- The optimizer consumes the exact cycle-\(t\) payload; the new score becomes selection evidence for cycle \(t+1\).

One update requires:

- exactly 32 distinct problem IDs;
- exactly eight rollout rows per problem;
- one identical policy/checkpoint version across all 256 rows;
- exact prompt, pseudo-gold, verifier, reward, sampling, and payload-size matches;
- no group copied twice and no modulo padding;
- no fallback generation if replay lookup fails.

An underfilled batch is a valid skipped cycle, not an invitation to train on stale, duplicated, label, or newly selected rollouts.

## Persistence and resume

Problem records and score observations are durable; opaque `DataProto` replay payloads are not durable and are never reused across cycles.

- Append candidate decisions, terminal label observations, accepted problems, score observations, and training events before updating derived indexes.
- Treat `map_index.json` as reconstructible from the accepted catalog.
- The engine invokes one external checkpoint after every successful optimizer update while the resident VERL fit thread is blocked at the exact post-weight-sync barrier. Only after that save does it release the next dataset fetch and commit the cycle boundary.
- `training.save_freq: 0` and `verl_config.trainer.save_freq: 0` disable periodic/scheduled saves. The external per-update checkpoint is unconditional and does not use either frequency value.
- A run stops at 256 successful optimizer updates or 320 total cycles, whichever comes first. A skipped cycle advances only the cycle counter.
- State and checkpoint markers identify the last committed policy boundary.
- On a crash before the optimizer commit, discard in-memory replay, record the interrupted cycle as `aborted`, advance to a fresh iteration namespace, and remeasure as needed under the restored checkpoint. Never regenerate stochastic text under the old candidate/label/score IDs.
- On a crash after an optimizer update but before its checkpoint/state commit, do not infer a usable policy from JSONL alone; restore the last committed checkpoint and quarantine policy-relative future observations.
- Resume must reject a changed schema, manifest, model identity, prompt/rules hash, or incompatible training geometry.

These rules favor an explicit skipped iteration over a silent on-policy violation.

## Required VERL sampling patch

Before preflight or run, apply the repository's idempotent environment patch:

```bash
python patches/verl_agent_loop_sampling.py
```

It makes the installed async agent-loop worker honor each `DataProto.meta_info` override for temperature, top-p, top-k, maximum tokens, log probabilities, and allowed token IDs. This is load-bearing for sharing one resident worker among crossover, nine-rollout labeling, eight-rollout scoring, and calibrated one-token domain classification. Preflight verifies the marker and fails before Ray reserves GPUs when the patch is absent. An unknown VERL source anchor is an error, not permission to continue unpatched.
