"""Frozen-config experiment runner; output is actual synthetic replay evidence."""
import hashlib
import json
from pathlib import Path
import platform
import statistics
from dataclasses import asdict
from .environment import Environment, make_tasks, schedule, ToolFailure
from .policies import run_policy
from .statistics import paired_summary


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2) + "\n"


def validate_config(config):
    if config["mode"] != "deterministic-synthetic-replay":
        raise ValueError("this runner supports replay only")
    if config["evaluation_split"] not in {"test", "tuning"}:
        raise ValueError("invalid split")
    if type(config["tool_call_budget"]) is not int or not 1 <= config["tool_call_budget"] <= 1000:
        raise ValueError("invalid budget")
    if not 0 <= config["fault_rate"] <= 1:
        raise ValueError("invalid fault rate")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    if config["policies"] not in (["no-retry", "fixed-retry", "checkpoint-budget"],
                                 ["no-retry", "fixed-retry", "step-retry", "checkpoint-budget"]):
        raise ValueError("original three or amended four policies required in protocol order")
    if type(config["checkpoint"]) is not bool or type(config["validate_outputs"]) is not bool:
        raise ValueError("ablation switches must be boolean")
    if config["bootstrap_repetitions"] != 2000:
        raise ValueError("v1 protocol fixes 2000 bootstrap repetitions")
    if config.get("task_generation", "legacy-v1") not in {"legacy-v1", "content-disjoint-v2"}:
        raise ValueError("unknown task generator")


def run(config_path: Path, output: Path):
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    # Refuse accidental result replacement. Config snapshot is persisted BEFORE any run.
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_bytes(config_bytes)
    digest = hashlib.sha256(config_bytes).hexdigest()
    (output / "preregistration.json").write_text(canonical({"config_sha256": digest,
        "state": "frozen before execution", "protocol": config["experiment_id"],
        "amendment": config.get("protocol_amendment")}), encoding="utf-8")
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(Path(__file__).parent.glob("*.py"))}
    (output / "source_manifest.json").write_text(canonical(source_hashes), encoding="utf-8")
    split = config["evaluation_split"]
    # Missing generation means the explicit historical protocol, so old configs retain
    # their original task identities/content. New protocols opt into the v2 partition.
    generation = config.get("task_generation", "legacy-v1")
    task_splits = {name: make_tasks(config["task_count_per_split"], name, generation)
                   for name in ("tuning", "test")}
    tasks = task_splits[split]
    overlap = set(task.increments for task in task_splits["tuning"]) & set(task.increments for task in task_splits["test"])
    if generation == "content-disjoint-v2" and overlap:
        raise AssertionError("v2 content split violation")
    manifests = {name: [task.public() for task in entries] for name, entries in task_splits.items()}
    (output / "tasks.json").write_text(canonical({"kind": "synthetic", "splits": manifests,
        "generation": generation, "cross_split_unique_content_overlap": len(overlap),
        "tuning_status": "not used to select this baseline configuration"}), encoding="utf-8")
    rows = []
    with (output / "traces.jsonl").open("w", encoding="utf-8") as trace:
        for task in tasks:
            for seed in config["seeds"]:
                faults = schedule(task.task_id, seed, config["tool_call_budget"], config["fault_rate"])
                for policy in config["policies"]:
                    env = Environment(task, faults)

                    def tool(action):
                        # Trace only structured safe metadata; raw tool/model text is never persisted.
                        event = {"task_id": task.task_id, "seed": seed, "policy": policy,
                                 "call_index": env.calls + 1,
                                 "operation": "observe" if action is None else "apply",
                                 "step": None if action is None else action.step}
                        try:
                            raw = env.call(action)
                            event["transport"] = "returned"
                            return raw
                        except ToolFailure:
                            event["transport"] = "failed"
                            raise
                        finally:
                            if env.last_attempt is not None:
                                event["attempt"] = asdict(env.last_attempt)
                            trace.write(json.dumps(event, sort_keys=True) + "\n")

                    row = run_policy(task.public(), tool, policy, config["tool_call_budget"],
                        config["fixed_retries"], config["checkpoint"], config["validate_outputs"])
                    row.update(seed=seed, split=split, success=env.oracle_success() and not row["abstained"],
                               goal_state_reached=env.oracle_success(), unsafe_proposed_actions=env.unsafe)
                    if row["tool_calls"] != env.calls:
                        raise AssertionError("budget accounting mismatch")
                    rows.append(row)
    summary = {}
    for policy in config["policies"]:
        selected = [row for row in rows if row["policy"] == policy]
        summary[policy] = {"runs": len(selected), "task_success": statistics.mean(row["success"] for row in selected),
            "abstention": statistics.mean(row["abstained"] for row in selected),
            "tool_calls_mean": statistics.mean(row["tool_calls"] for row in selected),
            "model_calls": sum(row["model_calls"] for row in selected),
            "unsafe_proposed_actions": sum(row["unsafe_proposed_actions"] for row in selected),
            "recovered_failures": sum(row["recovered_failures"] for row in selected),
            "failures": sum(row["failures"] for row in selected)}
    report = {"schema_version": 1, "mode": config["mode"], "dataset_kind": "owned synthetic closed-world ledger",
        "trace_schema_version": 2,
        "config_sha256": digest, "runtime": {"python": platform.python_version()}, "split": split,
        "task_generation": generation, "cross_split_unique_content_overlap": len(overlap),
        "latency": None, "latency_reason": "not measured; simulated call units are not time or money",
        "evidence_boundary": config["evidence_boundary"], "summary": summary, "runs": rows,
        "protocol_amendment": config.get("protocol_amendment"),
        "paired_comparisons": [paired_summary(rows, base, "checkpoint-budget")
                               for base in config["policies"] if base != "checkpoint-budget"],
        "limitations": ["Synthetic task-family uncertainty only; held-out IDs share the same generator family.",
            f"Task generator {generation}; cross-split shared unique task contents: {len(overlap)}. Historical IDs alone did not ensure content separation.",
            "No model has been run. Replay controllers are handwritten, not LLM agents.",
            "Fixed-retry restarts acknowledged work; checkpoint-budget preserves it. This engineering mechanism is not established novelty.",
            "Original restart-baseline comparisons are exploratory with an insufficient baseline. Amended step-retry retains progress under the same total budget.",
            "Recovery correctness counts failures followed by a validated acknowledged action; final success also requires evaluator state.",
            "Fault schedules align by call index across policies, not semantic operation; different trajectories see different operations faulted.",
            "No safety claim beyond closed-world action validation; no real filesystem or network tool behavior."]}
    (output / "report.json").write_text(canonical(report), encoding="utf-8")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())}
    (output / "SHA256SUMS.json").write_text(canonical(hashes), encoding="utf-8")
    return report
