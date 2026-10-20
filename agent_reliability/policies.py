"""Controllers receive only public task data and an opaque tool callable."""
from dataclasses import dataclass, asdict
from typing import Callable
from .contracts import Action, ContractError, observation
from .environment import ToolFailure


@dataclass
class Outcome:
    task_id: str
    policy: str
    tool_calls: int = 0
    model_calls: int = 0
    rejected_outputs: int = 0
    failures: int = 0
    recovered_failures: int = 0
    abstained: bool = True


def run_policy(public_task: dict, call: Callable, policy: str, budget: int,
               fixed_retries: int = 2, checkpoint: bool = True, validate: bool = True) -> dict:
    if policy not in {"no-retry", "fixed-retry", "step-retry", "checkpoint-budget"}:
        raise ValueError("unknown policy")
    result = Outcome(public_task["task_id"], policy)
    step = 0
    version = 0
    retries = 0
    pending_failures = 0
    state = None
    while result.tool_calls < budget and step < len(public_task["increments"]):
        action = None if state is None else Action(step, public_task["increments"][step], version)
        result.tool_calls += 1  # Debit before calling, including every failure.
        try:
            raw = call(action)
            if validate:
                response = observation(raw, public_task["task_id"], version)
            else:
                # Explicitly unsafe validation ablation; still never executes response text.
                import json
                response = json.loads(raw)
            if response["completed"] > len(public_task["increments"]):
                raise ContractError("impossible progress")
            if response["balance"] != sum(public_task["increments"][:response["completed"]]):
                raise ContractError("invalid balance")
            version = response["version"]
            state = response
            if action is not None:
                step += 1
                retries = 0
                result.recovered_failures += pending_failures
                pending_failures = 0
        except (ToolFailure, ContractError, ValueError, TypeError, KeyError) as exc:
            result.failures += 1
            pending_failures += 1
            if not isinstance(exc, ToolFailure):
                result.rejected_outputs += 1
            retries += 1
            if policy == "no-retry" or (policy == "fixed-retry" and retries > fixed_retries):
                break
            if policy == "fixed-retry" or (policy == "checkpoint-budget" and not checkpoint):
                step = 0
            # Conventional step-retry retains progress and shares the total budget.
            # Its equivalence to checkpoint-budget is an intended negative control.
            # A failed write may have taken effect: refresh state, never reuse stale version.
            state = None
            continue
        if action is None:
            # Restarted baselines replay prior steps idempotently; checkpoint tracks local progress.
            continue
    result.abstained = step != len(public_task["increments"])
    return asdict(result)
