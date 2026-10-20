"""Strict JSON wire contracts. Tool text is data and never executable instruction."""
from dataclasses import dataclass
from enum import Enum
import json


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Task:
    task_id: str
    increments: tuple[int, ...]
    split: str

    def public(self) -> dict:
        return {"task_id": self.task_id, "increments": list(self.increments)}


@dataclass(frozen=True)
class Action:
    step: int
    amount: int
    version: int


class AttemptOutcome(str, Enum):
    OBSERVATION = "observation_success"
    APPLIED = "action_applied"
    REPLAY = "idempotent_replay"
    INVALID = "invalid_action"
    VERSION_CONFLICT = "version_conflict"
    TRANSIENT = "transient_failure"


@dataclass(frozen=True)
class AttemptAudit:
    """Evaluator-side audit only; never included in a policy tool response."""
    outcome: AttemptOutcome
    version_before: int
    version_after: int
    unsafe_proposed: bool
    effect_applied: bool
    scheduled_fault: str | None
    response_fault: str | None


def strict_json(raw: str) -> dict:
    if not isinstance(raw, str) or len(raw) > 16384:
        raise ContractError("invalid response size/type")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ContractError("nonfinite number")))
    except (ValueError, TypeError, RecursionError) as exc:
        raise ContractError("invalid JSON") from exc
    if type(value) is not dict:
        raise ContractError("expected object")
    return value


def observation(raw: str, task_id: str, minimum_version: int) -> dict:
    value = strict_json(raw)
    if set(value) != {"task_id", "version", "completed", "balance"}:
        raise ContractError("unexpected tool fields")
    if value["task_id"] != task_id:
        raise ContractError("wrong task")
    if any(type(value[key]) is not int for key in ("version", "completed", "balance")):
        raise ContractError("noninteger observation")
    if value["version"] < minimum_version or value["completed"] < 0:
        raise ContractError("stale observation")
    if value["version"] != value["completed"]:
        raise ContractError("inconsistent observation")
    return value


def parse_action(raw: str, public_task: dict, version: int, step: int) -> Action:
    value = strict_json(raw)
    if set(value) != {"step", "amount", "version"}:
        raise ContractError("unexpected action fields")
    if any(type(value[key]) is not int for key in value):
        raise ContractError("action fields must be integers")
    if value != {"step": step, "amount": public_task["increments"][step], "version": version}:
        raise ContractError("action violates public task")
    return Action(**value)
