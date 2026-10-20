# EXP-001 post-exploratory amendment — 2026-10-20

The original restart-baseline result is exploratory and insufficient for attributing an advantage to checkpointing. Independent review requested a conventional baseline that retains acknowledged progress and retries within the same total budget. `step-retry` implements that baseline without receiving evaluator labels, hidden schedules or additional call resources. In this harness it is behaviorally equivalent to `checkpoint-budget`; there is no distinct research mechanism yet.

The amended configs were saved after the original outcomes and before their amended executions. Original artifacts were preserved. Previously inspected test IDs were reused for a paired engineering control. This is not a fresh confirmatory test. Tuning/test IDs are disjoint but their task content overlaps: tuning-0030 and test-0010 both have increments `(9, 5)`. No unseen-content or out-of-family generalization is established.

## Actual null result

`artifacts/amendment-strong-baseline/report.json` contains 800 runs: 40 task units × five seeds × four policies. Conventional step-retry and checkpoint-budget both achieved success 1.000, mean 7.415 tool calls, 283 failed calls across 200 task/seed runs, and zero unsafe proposals. Their paired success difference is exactly **0**, with the task-cluster bootstrap interval **[0, 0]**. Their equal result is preserved and is the current main finding. The amended fault-free control also has equal success across all four policies.

This invalidates an interpretation that the earlier +.210 gap to the weaker restart baseline demonstrates a new checkpoint method. The candidate remains useful engineering scaffolding. It needs a distinct, justified mechanism and stronger task protocol before a novel research project is supported.

## Safety-accounting repair

Independent review found that simulated transient failure was checked before invalid-action counting. An invalid proposed action coinciding with a transient fault could therefore be omitted from `unsafe_proposed_actions`. The environment now checks proposal validity first; attempted calls are still charged, and valid actions have no effect when transient transport fails. Two new regressions cover invalid-plus-transient and valid-plus-transient cases. All **18 tests passed** after this repair.

`artifacts/amendment-validation-accounting/report.json` is a separately frozen 800-run corrected validation-disabled replay on tuning IDs. Unsafe proposals are 22 for no-retry, 62 for restart fixed-retry, and 49 each for conventional step-retry and checkpoint-budget. The original validation-disabled checkpoint count 45 is retained as a historical undercount, superseded by 49. Apparent success remains 1.000 for step-retry/checkpoint; these unsafe attempts are not hidden or presented as a safety win.

## Exact rerun commands

```sh
python -m unittest discover -s tests -v
python -m agent_reliability run --config configs/amendment-strong-baseline.json --output artifacts/local-amended
python -m agent_reliability report --input artifacts/local-amended/report.json --output artifacts/local-amended/report.html
python -m agent_reliability run --config configs/amendment-null-control.json --output artifacts/local-amended-null
python -m agent_reliability run --config configs/amendment-validation-accounting.json --output artifacts/local-amended-validation
```

The actual execution used corresponding existing output names without `local-`: `amendment-strong-baseline`, `amendment-null-control`, and `amendment-validation-accounting`; report rendering was run for all three. Combined with the six original runs, **6,000 task/seed/policy runs** are retained. All are handwritten-controller synthetic replay with zero model calls. No real inference, GPU execution, external dataset, paid service or remote Git write occurred.

## SHA-256 locators

| Artifact | SHA-256 |
| --- | --- |
| `configs/amendment-strong-baseline.json` | `75dc8a5b4ede30ff61fec67269df5792a0de78c84abb2773180984e0ba245055` |
| `artifacts/amendment-strong-baseline/report.json` | `f4c3189f274177b490e63d8fd8fc3aec9c4cd8450464932edfc82169a2a14fbb` |
| `artifacts/amendment-strong-baseline/traces.jsonl` | `c296fb4c0913538529c273cad1238c966f01aeeda21698ddc135c74b8daf8b0b` |
| `artifacts/amendment-strong-baseline/report.html` | `07ec57408120d170c1f182f3ab3609efb42e92cf3d4c061bf259e35d88cf37ba` |
| `artifacts/amendment-validation-accounting/report.json` | `7bed0793bf54fe4216c5a43685e0e01aeb087bb0e553fdd5b6726c5ace2d60fe` |

Reports already exist; no run is still active. Browser visual review, real-model response integration, hosted-notebook execution and cross-version CI remain unverified. Independent review remains the authority for approval, not this author note.

## Content-disjoint and typed-audit replication — subsequent amendment

Review requested stronger content separation and per-attempt attribution. The original `legacy-v1` generator is retained explicitly for old configs (configs without `task_generation` continue using it). Their saved artifacts are unchanged. The public generator now defaults to `content-disjoint-v2`. It assigns a content sequence to tuning or test by the parity of a stable SHA-256 digest, then deterministically samples acceptable contents. Assignment uses no policy output, oracle label or fault outcome. The two partitions cannot accept the same content. Tests verify zero cross-split content overlap at the maximum supported 500 tasks per split, stable prefixes at different requested counts, and exact preservation of the historical `(9, 5)` overlap when legacy-v1 is selected.

This fixes cross-split content duplicates within v2; it does not make the synthetic family new, imply disjoint content relative to every historical v1 run, or prevent duplicate contents within a split. Task identities now include `v2`, so fault schedules are deterministic for the new IDs. Recovery policy logic, five seeds, 18-call budget, fault rate, metrics and task-unit bootstrap were not adjusted to outcomes. Hash partitioning changes the sampled task distribution; comparisons with historical absolute success rates are not an attribution experiment.

The new typed `AttemptAudit` records an enum outcome: observation success, newly applied action, idempotent replay, invalid action, version conflict, or transient failure. It includes pre/post versions, an explicit state-effect flag, unsafe-proposal flag, scheduled synthetic fault and actually emitted response corruption (these differ when an action is rejected before a scheduled corruption is emitted). `ToolFailure` exposes a typed failure category. The audit is evaluator-owned and serialized only to JSONL after the attempt; it is never attached to policy tool responses. Raw tool/model text remains excluded. Audit records describe environment effects and response corruption; they do not claim to record every controller reasoning step or client-validation decision.

`configs/amendment-content-disjoint-audit.json` was frozen before the new replication. Results are in `artifacts/amendment-content-disjoint-audit/`, including config receipt, task manifest with overlap count, typed JSONL trace, metrics, HTML/SVG, per-run hashes and new `source_manifest.json` hashes for the exact core source at execution. The replication has 800 task/seed/policy runs and **zero shared unique task contents between its tuning and test partitions**. Step-retry and checkpoint-budget both have success **.995**, mean **7.18 calls**, and **zero unsafe proposals**; their paired difference remains **0 with interval [0, 0]**. The single failed task/seed instance per strong policy is retained. All **21 targeted harness tests passed** after the change. The retained total is now **6,800 synthetic replay runs**.

Actual commands used the bundled Python 3.12.14 executable:

```sh
python -m unittest discover -s tests -p test_harness.py -v
python -m agent_reliability run --config configs/amendment-content-disjoint-audit.json --output artifacts/amendment-content-disjoint-audit
python -m agent_reliability report --input artifacts/amendment-content-disjoint-audit/report.json --output artifacts/amendment-content-disjoint-audit/report.html
```

New SHA-256 locators:

| Artifact | SHA-256 |
| --- | --- |
| `configs/amendment-content-disjoint-audit.json` | `5e17845986106d3e6bccc542a25272106730dcf270d053d8a25dfa70b90cc995` |
| `artifacts/amendment-content-disjoint-audit/report.json` | `67dd2abb8761613b539c99307661f7dc38a9df5e58d5dc83f7e9c22be69e3edb` |
| `artifacts/amendment-content-disjoint-audit/traces.jsonl` | `9b62bcda856f6a29cf06b4941cee9e603e722f0f6df12647e66e3d3b78ae97d5` |

No GPU pilot, notebook, README or CI file was edited in this followup; those are being integrated separately. No remote write or model execution occurred. Historical missing audit fields remain missing rather than being backfilled as though they were captured at the time.
