# First engineering slice — 2026-10-20

**Historical exploratory evidence; insufficient original baseline.** The subsequent [amendment](AMENDMENT.md) adds a strong conventional baseline and preserves a zero advantage. A reviewer also identified unsafe-proposal undercounting when transient failures coincided with invalid proposals; the validation-disabled results below are historical lower bounds, superseded by the corrected amendment run. Original artifact bytes are preserved.

Executed with bundled CPython 3.12.14 on Windows (exact runtime also recorded in each report). This is deterministic synthetic replay. No real model or hosted GPU was run. Mock HTTP tests do not prove real model behavior. No remote Git writes or package installs were performed.

## Commands actually run

Working directory: this repository. Here `python` refers to the bundled CPython 3.12.14 runtime, not the laptop's older default executable. The private machine-specific installation path is omitted; use any supported Python 3.11+ runtime to run the current harness.

```sh
python -m unittest discover -s tests -v
python -m agent_reliability run --config configs/experiment.json --output artifacts/experiment
python -m agent_reliability report --input artifacts/experiment/report.json --output artifacts/experiment/report.html
```

The same run/report pair was executed for each of `null-control`, `checkpoint-disabled`, `validation-disabled`, `budget-8`, and `budget-12`, replacing the config and output names. Every config existed before its execution. All six completed: 600 task/seed/policy runs each, 3,600 total. The test suite passed all 15 tests. Report rendering was checked by tests for saved real-JSON consumption and SVG creation; a human/browser visual review is pending.

## Observed results; preserve controls and limitations

| Config | Split | No-retry success | Fixed-retry success | Checkpoint-budget success | Checkpoint unsafe attempts |
| --- | --- | --- | --- | --- | --- |
| experiment | test | .310 | .790 | 1.000 | 0 |
| null-control | tuning | 1.000 | 1.000 | 1.000 | 0 |
| checkpoint-disabled | tuning | .350 | .745 | .870 | 0 |
| validation-disabled | tuning | .505 | .885 | 1.000 | 45 |
| budget-8 | tuning | .350 | .555 | .655 | 0 |
| budget-12 | tuning | .350 | .690 | .890 | 0 |

The main paired checkpoint-minus-fixed difference is +.210 with a task-cluster percentile interval [.150, .270], conditional on 40 synthetic task units and five seeds per task. The fault-free difference is exactly zero. Validation-disabled runs demonstrate that higher apparent success can coexist with unsafe action attempts. These are not claims about real agents, robustness outside the generator family or research novelty.

**Review priority:** the fixed-retry baseline restarts acknowledged workflow work. Checkpoint-budget preserves it and permits more consecutive retries within budget. Thus the comparison changes two recovery choices. A stronger fixed-step retry baseline retaining progress could reduce or eliminate the gap. Independent review should address this before research/publication claims. Ablations use tuning IDs and must not be compared numerically with main test as though they share identical tasks. Faults align by call index, not operation identity.

## Main artifact hashes (SHA-256)

| File | SHA-256 |
| --- | --- |
| `configs/experiment.json` | `2baee53b2e88feea2148ba6d51cd604e6a9b558a482adacb1c0470954f7bf791` |
| `artifacts/experiment/report.json` | `2f6cc9c3ad5d62f97a12d7db2fdf16a06e9a865e12286174f44cf0278153f50f` |
| `artifacts/experiment/traces.jsonl` | `6d7a284171a3769d13b57a695b02f9a09bd50fa89e65296985c60c0224d6e499` |
| `artifacts/experiment/report.html` | `19821e30d9b5c6422ee7c95ce1c3eace9efd1cd80f9b8080b5f39828b2bb48ff` |
| `artifacts/experiment/report.svg` | `6169d21ee88b45b362c6285842f958d4a38b5d4123419f7b356e8f88fb548813` |

Per-run `SHA256SUMS.json` covers execution artifacts; HTML/SVG are derived after that receipt. They are separately hashed above for the main run. Outputs total roughly 4.5 MB across six runs.

## Focused security inspection

Applied the local `security-scan` checklist to the new code. Static searches found no shell/process execution, dynamic evaluation, hardcoded credential or remote API target. The only network-capable module is an unused opt-in Ollama adapter, restricted to credential-free loopback HTTP origins with proxies and redirects disabled. Negative tests cover endpoint rejection, malicious action fields, stale/instruction-bearing output rejection, and exclusion of caller sidecar fields from mock model prompts. Raw tool/model content is not written to traces. All fixtures and code are original and synthetic; there was no external code copy requiring license reuse.

This is an author inspection, not independent approval or a complete penetration test. The code does not sandbox malicious Python callers. Real-model, cross-machine CI, browser layout and hosted notebook execution remain unverified.
