"""Injected model controller; no provider imports, network or evaluator access."""
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable

from .contracts import ContractError, observation, parse_action
from .environment import ToolFailure


@dataclass(frozen=True)
class Budgets:
    tool_calls: int
    model_calls: int
    tokens: int
    max_output_tokens: int = 96

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise ValueError("budgets must be nonnegative integers")
        if self.max_output_tokens == 0:
            raise ValueError("positive per-call output cap required")


@dataclass(frozen=True)
class ModelRequest:
    prompt: str
    max_output_tokens: int
    seed: int


@dataclass(frozen=True)
class ModelReply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    seed_applied: bool | None = None


@dataclass
class ModelOutcome:
    task_id: str
    policy: str = "model-step-retry"
    tool_calls: int = 0
    model_calls: int = 0
    token_debits: int = 0
    reported_input_tokens: int = 0
    reported_output_tokens: int = 0
    unknown_usage_calls: int = 0
    model_failures: int = 0
    model_rejections: int = 0
    tool_failures: int = 0
    rejected_tool_outputs: int = 0
    acknowledged_complete: bool = False
    abstained: bool = True
    budget_guarantee_valid: bool = True
    stop_reason: str = "not_started"


def _public_task(task: dict) -> dict:
    identity, increments = task.get("task_id"), task.get("increments")
    if (type(identity) is not str or not 1 <= len(identity) <= 128 or
            not all(character.isalnum() or character in "-_" for character in identity) or
            type(increments) not in (list, tuple) or not 1 <= len(increments) <= 100 or
            any(type(value) is not int or not 1 <= value <= 1000000 for value in increments)):
        raise ValueError("expected a bounded synthetic public task")
    return {"task_id": identity, "increments": list(increments)}


def run_model_policy(public_task: dict, tool: Callable, generate: Callable[[ModelRequest], ModelReply],
                     count_prompt_tokens: Callable[[str], int], budgets: Budgets,
                     seed: int, emit: Callable[[dict], None] | None = None) -> dict:
    """Return controller accounting only. Evaluator assigns goal/safety labels later.

    A reviewed provider must enforce max_output_tokens using the same tokenizer as
    count_prompt_tokens. Unknown usage is conservatively charged; a broken provider
    bound is retained as a violation rather than hidden behind the reservation.
    """
    task = _public_task(public_task)
    if type(seed) is not int:
        raise ValueError("integer seed required")
    emit = emit or (lambda event: None)
    result = ModelOutcome(task["task_id"])
    step, version, state = 0, 0, None

    def checked(raw):
        value = observation(raw, task["task_id"], version)
        progress = value["completed"]
        if progress > len(task["increments"]) or value["balance"] != sum(task["increments"][:progress]):
            raise ContractError("tool state violates public task")
        return value

    def invoke(action):
        result.tool_calls += 1
        event = {"kind": "tool", "attempt": result.tool_calls,
                 "action": None if action is None else asdict(action)}
        try:
            raw = tool(action)
        except ToolFailure as exc:
            result.tool_failures += 1
            event.update(outcome="tool_failure", failure_type=exc.outcome.value)
            emit(event)
            return None
        except Exception as exc:
            result.tool_failures += 1
            event.update(outcome="tool_exception", exception_type=type(exc).__name__)
            emit(event)
            return None
        event["raw_synthetic_tool_output"] = raw if isinstance(raw, str) else None
        try:
            value = checked(raw)
            if action is not None and value["completed"] != max(state["completed"], step + 1):
                raise ContractError("write acknowledgment did not match expected progress")
        except (ContractError, ValueError, TypeError, KeyError):
            result.rejected_tool_outputs += 1
            event["outcome"] = "rejected_tool_output"
            emit(event)
            return None
        event["outcome"] = "validated_tool_output"
        emit(event)
        return value

    while step < len(task["increments"]):
        if result.tool_calls >= budgets.tool_calls:
            result.stop_reason = "tool_budget"
            break
        if state is None:
            state = invoke(None)
            if state is None:
                continue
            version = state["version"]
            continue
        if result.model_calls >= budgets.model_calls:
            result.stop_reason = "model_budget"
            break
        prompt = json.dumps({"instruction": "Return only JSON integer fields step, amount, version for the requested task step.",
            "task": task, "observation": state, "step": step}, sort_keys=True)
        try:
            input_reservation = count_prompt_tokens(prompt)
            if type(input_reservation) is not int or input_reservation <= 0:
                raise ValueError("nonempty prompt requires a positive integer token count")
        except Exception as exc:
            result.stop_reason = "token_counter_failure"
            emit({"kind": "token_counter", "outcome": "failed", "exception_type": type(exc).__name__})
            break
        reservation = input_reservation + budgets.max_output_tokens
        if result.token_debits + reservation > budgets.tokens:
            result.stop_reason = "token_budget"
            break
        result.model_calls += 1
        result.token_debits += reservation  # Pre-debit all potentially consumed tokens.
        event = {"kind": "model", "attempt": result.model_calls, "step": step,
                 "prompt": prompt, "input_reservation": input_reservation,
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                 "output_cap": budgets.max_output_tokens, "reservation": reservation,
                 "seed": seed, "charged_tokens": reservation}
        emit({**event, "phase": "dispatch", "outcome": "dispatched"})
        event["phase"] = "result"
        try:
            reply = generate(ModelRequest(prompt, budgets.max_output_tokens, seed))
        except Exception as exc:
            result.model_failures += 1
            result.unknown_usage_calls += 1
            event.update(outcome="provider_failure", exception_type=type(exc).__name__, usage="unknown_full_reservation_charged")
            emit(event)
            continue
        if not isinstance(reply, ModelReply) or type(reply.text) is not str:
            result.model_failures += 1
            result.unknown_usage_calls += 1
            result.stop_reason = "provider_reply_contract"
            event.update(outcome="provider_reply_contract", response_type=type(reply).__name__)
            # Raw text from a wrongly wrapped response remains evidence, never an action.
            if isinstance(reply, str):
                event["raw_synthetic_model_output"] = reply
            emit(event)
            break
        event.update(raw_synthetic_model_output=reply.text, seed_applied=reply.seed_applied,
                     reported_input_tokens=reply.input_tokens, reported_output_tokens=reply.output_tokens)
        usage = (reply.input_tokens, reply.output_tokens)
        if usage == (None, None):
            result.unknown_usage_calls += 1
            event["usage"] = "unknown_full_reservation_charged"
        elif (any(type(value) is not int or value < 0 for value in usage) or
              reply.input_tokens != input_reservation or reply.output_tokens > budgets.max_output_tokens):
            result.budget_guarantee_valid = False
            result.stop_reason = "provider_usage_contract"
            event["outcome"] = "provider_usage_contract"
            emit(event)
            break
        else:
            result.reported_input_tokens += reply.input_tokens
            result.reported_output_tokens += reply.output_tokens
            charged = reply.input_tokens + reply.output_tokens
            result.token_debits -= reservation - charged
            event.update(charged_tokens=charged, usage="reported_within_reserved_bounds")
        try:
            action = parse_action(reply.text, task, version, step)
        except (ContractError, ValueError, TypeError):
            result.model_rejections += 1
            event["outcome"] = "rejected_model_output"
            emit(event)
            continue
        event.update(outcome="validated_model_action", action=asdict(action))
        emit(event)
        state = invoke(action)
        if state is None:
            continue  # Ambiguous write: observe before retrying the same idempotent step.
        version = state["version"]
        step += 1
    if step == len(task["increments"]):
        result.acknowledged_complete = True
        result.abstained = False
        result.stop_reason = "acknowledged_complete"
    return asdict(result)
