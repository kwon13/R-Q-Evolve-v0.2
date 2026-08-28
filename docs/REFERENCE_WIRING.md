# Reference wiring

R-Q-Evolve-v0.2 is a standalone implementation. The two neighboring repositories are **design references only**: no source file here imports `rq_evolve`, imports an R-Zero module, modifies `sys.path` to reach either repository, or shells out to their scripts.

## R-Zero ideas retained and tightened

| Design concern | Inspected reference | v0.2 behavior |
| --- | --- | --- |
| Free-form challenger output | `../R-Zero/question_generate/question_generate.py` | Extends the one `<question>...</question>` plus `\boxed{...}` envelope with one strict seven-value `<domain>` field; the prompt contains two concrete parents and each of 32 pair prompts requests two samples. |
| Independent pseudo-label rollouts | `../R-Zero/question_evaluate/evaluate.py` | Retains nine solver samples, but fixes the denominator at nine. Boxless or invalid outputs do not disappear from the confidence denominator. |
| Symbolic answer clustering | `../R-Zero/question_evaluate/evaluate.py` | Uses hard-timeout grading, checks equivalence symmetrically, requires a unique winning cluster, and abstains on non-transitive or ambiguous relations. |
| Challenger answer | Generated-question JSON in R-Zero | Retains it as generation metadata. As in the inspected evaluator, Solver consensus supplies pseudo-gold and a Challenger-answer mismatch does not reject the question. |

R-Zero's original evaluator computes agreement over only extracted answers. v0.2 follows its non-gating treatment of the proposed answer, while deliberately keeping the stricter fixed denominator of nine.

## R-Q-Evolve ideas retained and tightened

| Design concern | Inspected reference | v0.2 behavior |
| --- | --- | --- |
| Policy-relative learnability | `../R-Q-Evolve/src/rq_evolve/scoring.py` and `evolution.py` | Computes a current-policy success rate, unbiased Bernoulli learnability, normalized response entropy, and \(R_Q=L\times U\). |
| One-step delayed selection | `../R-Q-Evolve/src/rq_evolve/replay.py` | Uses the score known before cycle \(t\) to choose eligible incumbents; a problem first measured at \(t\) cannot train until \(t+1\). |
| Exact rollout reuse | `../R-Q-Evolve/src/rq_evolve/verl_backend.py`, `replay.py`, and `replay_hook.py` | Retains the actual current-policy `DataProto` slice for each complete score group and serves it to the trainer without another generation call. |
| Live training integration | `../R-Q-Evolve/src/rq_evolve/verl_adapter.py` and `scripts/train_with_verl.py` | Uses the same long-lived actor/rollout worker and checkpoint lineage for evolution, scoring, and updates. |
| Descriptor MAP | `../R-Q-Evolve/src/rq_evolve/archive.py` | Uses the 7 top-level domains × 5 computational problem types, but stores all accepted problems per cell instead of only one champion. |

The replay contract is deliberately stricter than the inspected implementation. A missing group, policy mismatch, prompt mismatch, verifier mismatch, reward mismatch, wrong payload size, or short batch aborts/skips the update. It never silently falls through to fresh training generation.

The lag is also strict about which measurement controls selection: only cycle \(t-1\)'s score establishes eligibility and priority. Cycle \(t\)'s success rate does not re-filter the chosen problem; its complete current-policy payload is replayed even if all eight answers are correct or all eight are wrong.

## What is newly owned here

The following are v0.2 contracts rather than runtime dependencies on either reference:

- two-parent problem crossover rather than program mutation;
- exactly 32 pair prompts with two children each for the initial generation budget;
- fixed-denominator pseudo-label confidence with proposed-answer agreement retained only for audit;
- an append-only all-accepted problem catalog and a rebuildable MAP index;
- typed, data-only verifier contracts for expression, Boolean, and finite-set answers;
- strict separation of label rollouts, score observations, and replay payloads;
- a durable `label_observations.jsonl` containing terminal accepted and rejected pseudo-label evidence;
- exact 32-group, same-policy training batches with no modulo padding;
- explicit skip events for discovery-only or underfilled cycles;
- one pipeline-orchestrated checkpoint after every successful update, with VERL's periodic saver disabled.

Paths in this document record provenance for maintainers. They are not required to exist on a deployment server after this repository has been installed.
