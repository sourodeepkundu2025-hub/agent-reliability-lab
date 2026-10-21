# Model-controller protocol MP-001 — frozen design, 2026-10-21

Status: implementation and mock correctness validation only. No model execution, GPU allocation, checkpoint download or scientific result is asserted. Actual inference is restricted to a separately reviewed owned provider on available free Colab/Kaggle; local Ollama is not an execution route for this sprint.

## Question and mechanism

Can a model-generated action controller satisfy strict synthetic task contracts and recover ambiguous tool writes within separate model, tool and token budgets? This is a closed-world schema/recovery integration question, not a novel method or general planning benchmark. Every legal next action is determined by the public task and current observation. Deterministic step-retry is therefore a strong baseline, and model inference can introduce overhead or errors without any task-quality improvement. Preserve that null/degradation if observed.

`run_model_policy` receives a whitelisted public task, opaque tool callable, model-generation callback, prompt-token counter, immutable budget settings, seed and event sink. It receives no environment instance, evaluator success label, fault schedule, split name or hidden target. The model callback receives only an immutable JSON prompt, maximum output-token limit and seed. The prompt contains the synthetic public task, validated observation and next step. Tool text is validated before being inserted; rejected/instruction-bearing text never reaches the model. This is API separation, not a sandbox against malicious Python introspection.

The controller requests an observation, asks the model for exactly `{step, amount, version}`, validates with the existing strict parser, then invokes the tool. Malformed model responses are retained and rejected without tool execution. After any rejected tool response or tool failure, the controller refreshes its observation and retries the same idempotent step. A corrupted write may have taken effect; the refreshed version permits a duplicate-safe replay. Only a validated action acknowledgment advances local progress. Final success is assigned by the evaluator, not the policy.

## Frozen comparison

- Generator: `content-disjoint-v2`, 40 tuning IDs and 40 test IDs; task contents are partitioned by hash independent of policy outcomes. Within-split duplicate contents and overlap with historical v1 tasks remain possible. This is the same synthetic task family.
- Seeds: 11, 23, 47, 71, 101. Same deterministic call-index fault schedule per task/seed for all controllers: transient, malformed, stale and instruction-bearing output; fault probability .2. Faults are evaluator-only. Different trajectories can fault different semantic operations.
- Tool budget: 18 attempts for each policy, including failures. Strong baseline: existing deterministic `step-retry`, preserving acknowledged progress. Candidate: `model-step-retry`. No old controllers/results are modified.
- Model budget: 12 generation attempts, including provider exceptions and malformed outputs. Deterministic baseline uses zero model calls. These are separate resource dimensions, not falsely equal inference resources.
- Token budget: 12,000 total conservative token debits; at most 96 generated tokens per model attempt. Before inference, the reviewed provider's matching tokenizer must count the exact rendered prompt and reserve input tokens plus the output cap. If the remaining budget cannot cover the entire reservation, no call is dispatched. Failed or unknown-usage calls are charged the full reservation. Known valid usage can release unused output reservation only if the reported input count matches the tokenizer count exactly and output usage is a nonnegative integer within the cap. Underreported input usage is rejected too. Report reported input/output counts, conservative debits, unknown-usage calls and provider-bound violations separately. These are not billed cost. A provider that violates the declared prompt/output bound invalidates the budget guarantee and stops the controller; the breach is retained.
- Model/checkpoint/tokenizer revision, provider source hash, sampling parameters, seed application and actual hardware must be fixed by root's hosted integration before real inference. An unselected checkpoint is an execution gate, not a reason to invent a model run.
- Task success requires correct evaluator state and acknowledged controller completion. Abstention means no acknowledged completion before a budget/contract stop. Goal-state reached without acknowledgment is recorded separately. Invalid model proposals rejected by the guard, unsafe proposals reaching the environment, failures and ambiguous-write recoveries remain visible.
- Report all task/seed runs. Paired success differences average seed repeats inside task units and use the existing 2,000-resample task-cluster bootstrap. Conditioning on this synthetic family is explicit; multiple model calls are not independent task units.

## Records, immutability and stop criteria

The new runner creates a previously absent output directory and writes exclusive config/source/task/fault identities before dispatch. Provider identity and owned provider-source hashes are frozen too. Model JSONL preserves the exact raw synthetic returned text, malformed outputs, per-attempt counts/reservations/usage and parser outcome; exceptions record type without potentially secret exception strings. Tool JSONL includes typed action intent, safe environment audit, versions/effects and outcome. A final receipt hashes every execution artifact. Raw records are synthetic-only; no employer/private input is allowed. Raw model outputs are untrusted data and are never executed.

`--mock-validation` uses an owned deterministic callback and explicitly approximate mock token units; its output is fixture correctness evidence, never LLM inference. Its evidence label is retained in config, provider identity and report. CLI real inference is intentionally gated: root integrates a reviewed provider by calling the Python runner API, not arbitrary dynamic module execution.

Null/kill criteria: no claim of model advantage if step-retry matches or exceeds task success; reject a candidate if unsafe actions increase, malformed/unknown outputs are omitted, usage cannot be bounded, evaluator information enters policy input, or a comparison changes the fixed task/budget/fault protocol without amendment. Stop on any exhausted resource, malformed provider usage contract, unsupported execution identity or immutable-output collision. A mock pass does not authorize a hosted run, promotion or public result.

## Provider integration handoff

The new stdlib API is `run_model_policy(public_task, tool, generate, count_prompt_tokens, budgets, seed, emit)`. `generate` takes a frozen `ModelRequest(prompt, max_output_tokens, seed)` and returns `ModelReply(text, input_tokens, output_tokens, seed_applied)`. `count_prompt_tokens` must use the exact same rendered input/tokenizer as generation, including any provider chat template or special tokens. The dispatch prompt hash identifies the public JSON prompt before provider templating; root's provider source, immutable tokenizer/checkpoint revision and chat-template identity must make the actual rendered input reconstructable. `seed_applied` records what the provider really did rather than assuming the request seed was effective.

The runner API is `run_experiment(config_path, output, generate, count_prompt_tokens, provider_identity=..., provider_sources=...)`. It imports no external provider. Real-hosted identity requires free Colab/Kaggle location, explicit checkpoint/tokenizer revisions, hardware, sampling settings and owned provider-source paths. Those metadata are declared provenance, not a hardware attestation; root verifies actual hosted execution separately. Sources, cases, hidden fault schedule, provider metadata and config are frozen in separate exclusive files, hashed by `initial_receipt.json` before dispatch. Every completed case appends its metrics to flushed JSONL. Interrupted executions retain the initial receipt plus `failure_receipt.json` hashing partial artifacts; completed runs add a new completion receipt without rewriting initial records.

Mock-only command (new output directory required):

```sh
python -m unittest discover -s tests -p test_model_policy.py -v
python scripts/run_model_recovery.py --config configs/model-policy-mock-validation-v1.json --output path/to/new-private-mock-run --mock-validation
```

`configs/model-policy-v1.json` specifies the future real comparison, with checkpoint still unselected; root must freeze actual provider identities before any hosted call. The smaller `model-policy-mock-validation-v1.json` is a separate tuning-only correctness fixture, not a replacement test experiment. Neither code presence nor mock completion passes P02's actual-hosted-execution gate.
