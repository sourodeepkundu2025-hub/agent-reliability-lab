"""Frozen-record runner for an injected reviewed provider; CLI offers mocks only."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_reliability.environment import Environment, make_tasks, schedule
from agent_reliability.model_policy import Budgets, ModelReply, run_model_policy
from agent_reliability.policies import run_policy
from agent_reliability.statistics import paired_summary


def canonical(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def exclusive_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value))


def source_manifest(paths):
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def validate(config):
    if config["task_generation"] != "content-disjoint-v2":
        raise ValueError("content-disjoint generator required")
    if config["policies"] != ["step-retry", "model-step-retry"]:
        raise ValueError("frozen strong comparator required")
    if config["evaluation_split"] not in {"tuning", "test"}:
        raise ValueError("invalid split")
    if (type(config["task_count_per_split"]) is not int or not 1 <= config["task_count_per_split"] <= 500 or
        type(config["fault_rate"]) not in (int, float) or not 0 <= config["fault_rate"] <= 1 or
        not config["seeds"] or any(type(seed) is not int for seed in config["seeds"]) or
        len(set(config["seeds"])) != len(config["seeds"])):
        raise ValueError("invalid dataset/fault/seed protocol")
    return Budgets(config["tool_call_budget"], config["model_call_budget"],
                   config["token_budget"], config["max_output_tokens"])


def run_experiment(config_path, output, generate, count_prompt_tokens, *, provider_identity,
                   provider_sources=()):
    """Root hosted integration calls this API with an owned, reviewed adapter.

    No arbitrary provider import, model download or inference is performed by this
    module unless the caller explicitly supplies and invokes the provider callback.
    """
    config_path, output = Path(config_path), Path(output)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    budgets = validate(config)
    mode = provider_identity.get("evidence_mode")
    if mode not in {"unit-mock-validation", "real-free-hosted-model"}:
        raise ValueError("explicit mock or real hosted evidence identity required")
    if mode == "unit-mock-validation" and config.get("execution_mode") != mode:
        raise ValueError("mock mode requires a frozen explicitly mock config")
    if mode == "real-free-hosted-model":
        if (provider_identity.get("execution_location") not in {"free-colab", "free-kaggle"} or
            any(not provider_identity.get(key) for key in
                ("model_id", "checkpoint_revision", "tokenizer_revision", "hardware", "sampling_parameters")) or
            not provider_sources):
            raise ValueError("reviewed hosted provider/checkpoint/tokenizer/hardware/source identities required")
    sources = list((ROOT / "agent_reliability").glob("*.py")) + [Path(__file__)] + [Path(p) for p in provider_sources]
    sources_before = source_manifest(sources)
    splits = {split: make_tasks(config["task_count_per_split"], split, config["task_generation"])
              for split in ("tuning", "test")}
    cases = {split: [task.public() for task in tasks] for split, tasks in splits.items()}
    faults = {f"{task.task_id}:{seed}": schedule(task.task_id, seed, budgets.tool_calls, config["fault_rate"])
              for task in splits[config["evaluation_split"]] for seed in config["seeds"]}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "config.json").open("xb") as stream:
        stream.write(config_bytes)
    exclusive_json(output / "sources.json", sources_before)
    exclusive_json(output / "cases.json", cases)
    exclusive_json(output / "faults.json", faults)
    exclusive_json(output / "provider.json", provider_identity)
    identities = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())}
    # Persisted before the first provider call; never rewritten even if execution fails.
    exclusive_json(output / "initial_receipt.json", {"status": "started_not_complete", "evidence_mode": mode,
        "frozen_identities": identities, "python": platform.python_version()})
    rows = []
    try:
        with (output / "events.jsonl").open("x", encoding="utf-8") as stream:
            def write(event):
                stream.write(json.dumps(event, sort_keys=True) + "\n")
                stream.flush()  # Dispatch evidence exists before the external callback.

            for task in splits[config["evaluation_split"]]:
                for seed in config["seeds"]:
                    task_faults = faults[f"{task.task_id}:{seed}"]
                    for policy in config["policies"]:
                        environment = Environment(task, task_faults)
                        common = {"task_id": task.task_id, "seed": seed, "policy": policy, "evidence_mode": mode}

                        def opaque_tool(action):
                            try:
                                return environment.call(action)
                            finally:
                                write({**common, "kind": "environment_audit", "call_index": environment.calls,
                                    "action": None if action is None else asdict(action),
                                    "audit": None if environment.last_attempt is None else asdict(environment.last_attempt)})

                        if policy == "step-retry":
                            row = run_policy(task.public(), opaque_tool, policy, budgets.tool_calls)
                            row.update(acknowledged_complete=not row["abstained"], token_debits=0,
                                       budget_guarantee_valid=True, model_rejections=0)
                        else:
                            row = run_model_policy(task.public(), opaque_tool, generate, count_prompt_tokens,
                                budgets, seed, emit=lambda event: write({**common, **event}))
                        row.update(seed=seed, split=config["evaluation_split"],
                            goal_state_reached=environment.oracle_success(),
                            success=environment.oracle_success() and row["acknowledged_complete"],
                            unsafe_proposed_actions=environment.unsafe)
                        if row["tool_calls"] != environment.calls:
                            raise AssertionError("tool budget accounting mismatch")
                        rows.append(row)
                        write({**common, "kind": "completed_case_metrics", "metrics": row})
        if source_manifest(sources) != sources_before:
            raise RuntimeError("frozen source changed during execution")
        comparison = paired_summary(rows, "step-retry", "model-step-retry")
        report = {"schema_version": 1, "protocol_id": config["protocol_id"], "evidence_mode": mode,
            "evidence_boundary": ("Unit mock validation only; no LLM/model inference occurred." if mode == "unit-mock-validation"
                else "Actual provider execution as identified; synthetic schema/recovery task only."),
            "config_sha256": identities["config.json"], "provider_identity": provider_identity,
            "runs": rows, "paired_comparison": comparison,
            "compute_boundary": "Matched tool-call caps only; deterministic baseline has zero model/token compute. Not an equal-total-compute improvement comparison.",
            "token_boundary": "Verified matching prompt count plus provider-reported output within cap; failed/unknown calls charged full conservative reservations. Not billed cost.",
            "null_interpretation": "No model advantage if paired difference is zero or negative; retain all malformed outputs and abstentions.",
            "limitations": ["Same synthetic task family; no general agent-planning or novelty claim.",
                "No real hosted inference follows from mock validation or code presence.",
                "Sources/cases/faults/provider/config frozen before calls; dynamic prompt hash persisted at dispatch.",
                "Provider must enforce max output tokens with the same tokenizer; a breach invalidates budget guarantees."]}
        exclusive_json(output / "report.json", report)
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())}
        exclusive_json(output / "completion_receipt.json", {"status": "complete", "evidence_mode": mode,
            "task_seed_policy_runs": len(rows), "hashes": hashes})
        return report
    except BaseException as exc:
        partial_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(output.iterdir()) if path.is_file()}
        exclusive_json(output / "failure_receipt.json", {"status": "failed", "exception_type": type(exc).__name__,
            "completed_task_seed_policy_runs": len(rows), "evidence_mode": mode, "partial_artifact_hashes": partial_hashes})
        raise


def mock_generate(request):
    """Owned fixture, NOT inference: deterministic legal action from the public prompt."""
    value = json.loads(request.prompt)
    step = value["step"]
    text = json.dumps({"step": step, "amount": value["task"]["increments"][step],
                       "version": value["observation"]["version"]})
    return ModelReply(text, mock_token_count(request.prompt), mock_token_count(text), True)


def mock_token_count(text):
    """Artificial test units, not any real model's tokenizer."""
    return max(1, (len(text.encode()) + 3) // 4)


def main():
    parser = argparse.ArgumentParser(description="Model-policy correctness runner; CLI runs owned mocks only")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mock-validation", action="store_true", required=True)
    args = parser.parse_args()
    report = run_experiment(args.config, args.output, mock_generate, mock_token_count,
        provider_identity={"evidence_mode": "unit-mock-validation", "provider": "owned deterministic fixture",
                           "token_units": "artificial UTF-8-length units; not real tokenizer tokens"})
    print("MOCK VALIDATION ONLY; no LLM inference:", len(report["runs"]), "synthetic runs")


if __name__ == "__main__":
    main()
