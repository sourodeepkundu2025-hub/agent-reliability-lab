# Agent Reliability Lab

A small, runnable research-engineering baseline for studying recovery from tool failures under matched call budgets. **All checked-in results are deterministic synthetic replay, not LLM measurements.** Scientific novelty has not been established.

**Current finding:** checkpoint-budget matches a conventional step-retry baseline exactly in the amended synthetic test: success difference 0, task-cluster interval [0, 0], equal mean tool calls. Both controllers currently implement the same recovery mechanism. The earlier restart-baseline advantage is preserved as exploratory evidence with an insufficient baseline, not a scientific benefit. See [amendment evidence](docs/AMENDMENT.md).

## Run locally

Python 3.11+; standard library only. From this directory:

```sh
python -m unittest discover -s tests -v
python -m agent_reliability run --config configs/amendment-content-disjoint-audit.json --output artifacts/local-first
python -m agent_reliability report --input artifacts/local-first/report.json --output artifacts/local-first/report.html
```

Open the generated HTML locally. It renders the actual report JSON into an accessible table and SVG chart, with text values and evidence boundaries. The runner refuses to overwrite existing output directories. No package install, network call, GPU, credential or model download is needed. A configured hosted GPU notebook is a separate integration, not a requirement for this replay harness.

## What is implemented

The closed-world environment performs a synthetic ledger task: apply a public sequence of positive increments exactly once. This is stateful: every accepted new action changes the balance and version. Each step has an idempotency identity, so an unacknowledged write can be retried without doubling it. There is no real filesystem or external tool execution.

The no-retry baseline stops after any rejected response or tool failure. The fixed-retry baseline restarts acknowledged work after failure, with two retries between acknowledged actions. The checkpoint-budget variant retains acknowledged progress and keeps recovering until the shared budget is exhausted. A refreshed observation is required after failure because a corrupted write response may hide a completed effect. All attempted tool calls, including transient failures, malformed responses and rejected actions, are charged before execution. All runs have zero model calls.

The post-exploratory amendment adds **step-retry**, a stronger conventional baseline retaining acknowledged progress and retrying within the identical total call budget. Its behavior is currently equivalent to checkpoint-budget by design; preserving the null result prevents a weak restart baseline from creating a novelty claim. The amendment was frozen after the initial results and before its own execution. It reuses already inspected test IDs and is an engineering control, not a fresh confirmatory study.

Strict JSON contracts reject duplicate keys, nonfinite numbers, unexpected fields, noninteger fields, wrong task IDs, stale versions, inconsistent progress and wrong action values. Instruction-bearing tool text is rejected as an unexpected field. No tool text is evaluated or executed. `checkpoint` and `validate_outputs` are explicit ablation switches. Disabling output validation is an intentionally unsafe experimental ablation; it can increase rejected action attempts.

The evaluator creates fault schedules from stable task/seed identities and owns goal-state labels. The policy receives only the public task and an opaque tool callable; its interface has no fault schedule, oracle label or environment object. This is an architectural separation, not a sandbox against malicious Python introspection. Public increments describe the requested goal and are not hidden answers.

## Dataset and protocol

Every task is an original synthetic fixture; no public benchmark, private input or employer source is used. Historical v1 configs have disjoint IDs but overlapping content: tuning-0030 and test-0010 both request `(9, 5)`. They remain preserved as historical evidence. The subsequent `content-disjoint-v2` generator partitions contents by stable hashes, with tests for zero cross-split overlap across 500 tasks per split. This does not establish out-of-family generalization or a fresh unseen research benchmark. The latest frozen replication retains 800 runs with equal step-retry/checkpoint success .995 and paired difference 0. See [the full amendments](docs/AMENDMENT.md). Future changes must use tuning IDs and preserve a fresh protocol; repeated test inspection removes the clean hold-out interpretation.

`configs/experiment.json` preregisters the question, five seeds, all three policies, 18 tool calls per run, 20% fault probability, ablations and kill criteria. Before any execution the runner persists an exact config snapshot and hash receipt. For each task and seed all policies receive the same call-index fault schedule. Their different trajectories mean faults can affect different semantic operations; this confound is recorded in the report.

The main success metric requires both the evaluator's correct final state and controller acknowledgment. `goal_state_reached` separately retains successful writes whose response was rejected before acknowledgment. Abstention is not silently scored as success. Unsafe proposed actions count invalid actions reaching the environment, including failures in the validation-disabled ablation. Recovery counts previous failed calls followed by a validated action; it is an operational count, not a causal correctness proof. Latency and monetary cost are unmeasured, recorded as unavailable, and not inferred from call counts.

Paired differences average the five seed outcomes within each task, then resample task units with replacement 2,000 times. Percentile intervals are conditional on the synthetic task family. Repeated seeds are not treated as independent tasks. Preserve null and negative comparisons.

## Architecture and artifacts

| Module | Responsibility |
| --- | --- |
| `contracts.py` | Public task/action types and strict JSON parsing |
| `environment.py` | Synthetic mutable state, evaluator labels, seeded faults |
| `policies.py` | No-retry, restart fixed-retry, conventional step-retry and checkpoint-budget controllers |
| `experiment.py` | Freeze config, execute matched runs, persist traces and metrics |
| `statistics.py` | Paired task-cluster uncertainty |
| `report.py` | Render saved report JSON as HTML and standalone SVG |
| `ollama.py` | Optional loopback HTTP adapter and real Ollama response parser |

Each run saves `config.json`, `preregistration.json`, `tasks.json`, `traces.jsonl`, `report.json`, and `SHA256SUMS.json`; newer runs also record source hashes. The manifest hashes execution artifacts present at run completion; HTML/SVG are derived afterward and not covered by that manifest. New JSONL traces contain typed attempt outcomes, pre/post versions, state effects, unsafe proposals and scheduled versus emitted faults. Historical traces retain their originally captured fields. Raw tool/model text is deliberately excluded from this handwritten-controller replay. Report JSON contains every run, aggregates, comparisons and limitations.

Git text normalization is disabled for this repository so checked-in byte hashes survive Windows/Linux clones. Preserve these evidence bytes; if a formatter changes them, retain the original run and publish a separately identified derived artifact instead of silently rehashing history.

## Free hosted GPU pilot — prepared, not executed

[Open in Colab](https://colab.research.google.com/github/sourodeepkundu2025-hub/agent-reliability-lab/blob/main/notebooks/Colab_GPU_Pilot.ipynb) · [Kaggle notebook source](notebooks/Kaggle_GPU_Pilot.ipynb).

On Kaggle, create a notebook and import that `.ipynb`; enable Internet and a GPU if your account has allocation. In either provider, inspect all cells, then replace `REVIEWED_COMMIT` with the exact 40-character commit you reviewed from this repository. The guard deliberately refuses the placeholder. Dependencies are installed only inside your hosted runtime; the existing PyTorch version is recorded rather than silently replaced.

This opt-in pilot runs the pinned public `Qwen/Qwen3-0.6B` checkpoint on eight synthetic JSON-action cases. It records raw invalid outputs, package/model/source identity, the actual GPU, dtype, explicit eager attention, token counts and synchronized timings. It refuses CPU execution and existing output directories; interrupted sessions remain separate evidence folders. No account token, MCP or ngrok tunnel is required. No hosted GPU or model execution has yet been verified. It is **not** the multi-step agent-recovery experiment, a training run, or a measured improvement. Read [the model card](https://huggingface.co/Qwen/Qwen3-0.6B) and [the pilot source](scripts/gpu_pilot.py) before downloading weights.

## Optional local model adapter — not integrated or run

`OllamaAdapter(model="explicit-installed-checkpoint")` implements an HTTP POST to a loopback `/api/generate`, parses the actual Ollama `done`/`response` envelope, and validates the returned action. It never reads credentials, ignores proxies and refuses redirects or non-loopback origins. Call counts include failed attempts. Extra caller sidecar fields are excluded from prompts; whitelisted task values themselves must still be synthetic/nonprivate. Unit tests inject an in-memory mock transport: **these tests are not real model inference**. The replay runner never instantiates this adapter. No model has been downloaded or called. A real agent comparison needs an integrated model controller, matched model/tool budgets and a separately preregistered checkpoint/dataset protocol.

## Provenance

All code and fixture logic was originally authored for this repository; no external code was copied or executed. The starting method idea is the owned EXP-001 contract: baseline parity, checkpoint recovery, output validation, budget sweeps, paired uncertainty and explicit abstention. No paper reproduction or published comparison is claimed. A primary-source literature ledger and nearest-method review must precede any novelty claim.

Configurations, source identities and saved outputs document the implemented experiments. The checked-in model adapters and notebook scaffolding are not measured model results.
