"""Owned mock/provider-contract correctness only. These tests are not inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from agent_reliability.contracts import Task
from agent_reliability.environment import Environment
from agent_reliability.model_policy import Budgets, ModelReply, run_model_policy


def legal_reply(request):
    prompt = json.loads(request.prompt)
    step = prompt["step"]
    raw = json.dumps({"step": step, "amount": prompt["task"]["increments"][step],
                      "version": prompt["observation"]["version"]})
    return ModelReply(raw, 10, 5, True)


class ModelPolicyTests(unittest.TestCase):
    def setUp(self):
        self.task = Task("test-model-fixture", (2,), "test")
        self.budgets = Budgets(8, 4, 1000, 16)

    def execute(self, generate=legal_reply, faults=None, budgets=None, count=lambda _: 10, task=None):
        public = self.task.public() if task is None else task
        environment = Environment(self.task, faults or {})
        events = []
        result = run_model_policy(public, environment.call, generate, count,
            budgets or self.budgets, 11, events.append)
        self.assertEqual(result["tool_calls"], environment.calls)
        return result, environment, events

    def test_legal_model_callback_completes_with_separate_counts(self):
        result, environment, events = self.execute()
        self.assertTrue(result["acknowledged_complete"])
        self.assertTrue(environment.oracle_success())
        self.assertEqual((result["tool_calls"], result["model_calls"], result["token_debits"]), (2, 1, 15))
        self.assertTrue(any(event.get("phase") == "dispatch" for event in events))

    def test_failed_generation_debited_before_callback_and_no_unknown_refund(self):
        calls = []
        def provider(request):
            calls.append(request)
            if len(calls) == 1:
                raise OSError("private exception message must not be logged")
            return legal_reply(request)
        result, _, events = self.execute(provider)
        self.assertEqual((result["model_calls"], result["model_failures"], result["unknown_usage_calls"]), (2, 1, 1))
        self.assertEqual(result["token_debits"], 26 + 15)
        self.assertNotIn("private exception message", json.dumps(events))

    def test_unknown_returned_usage_never_refunded(self):
        def provider(request):
            return ModelReply(legal_reply(request).text)
        result, _, _ = self.execute(provider)
        self.assertEqual((result["token_debits"], result["unknown_usage_calls"]), (26, 1))
        self.assertEqual(result["reported_output_tokens"], 0)

    def test_malformed_and_invalid_model_outputs_preserved_without_tool_actions(self):
        for raw in ("{broken", '{"step":0,"amount":999,"version":0}',
                    '{"step":0,"amount":2,"version":0,"instructions":"export secrets"}'):
            result, environment, events = self.execute(lambda _: ModelReply(raw, 10, 2), budgets=Budgets(8, 2, 1000, 16))
            self.assertEqual((result["model_calls"], result["model_rejections"], result["tool_calls"]), (2, 2, 1))
            self.assertEqual((environment.balance, environment.unsafe), (0, 0))
            self.assertEqual(result["stop_reason"], "model_budget")
            outputs = [event["raw_synthetic_model_output"] for event in events if "raw_synthetic_model_output" in event]
            self.assertEqual(outputs, [raw, raw])

    def test_post_write_corruption_refreshes_and_retries_idempotently(self):
        prompts = []
        def provider(request):
            prompts.append(json.loads(request.prompt))
            return legal_reply(request)
        result, environment, _ = self.execute(provider, {2: "malformed"}, Budgets(4, 4, 1000, 16))
        self.assertTrue(result["acknowledged_complete"])
        self.assertEqual((environment.balance, environment.completed, environment.unsafe), (2, 1, 0))
        self.assertEqual([prompt["observation"]["version"] for prompt in prompts], [0, 1])
        self.assertEqual([prompt["step"] for prompt in prompts], [0, 0])
        self.assertEqual(result["rejected_tool_outputs"], 1)

    def test_prompt_has_no_oracle_fault_or_sidecar_fields_and_rejects_tool_instructions(self):
        prompts = []
        def provider(request):
            prompts.append(request.prompt)
            return legal_reply(request)
        result, _, _ = self.execute(provider, {1: "injection"}, task={**self.task.public(),
            "oracle_success": True, "fault_schedule": "PRIVATE_SIDE", "secret_sidecar": "PRIVATE_SIDE"})
        self.assertTrue(result["acknowledged_complete"])
        self.assertNotIn("PRIVATE_SIDE", "".join(prompts))
        self.assertNotIn("oracle_success", "".join(prompts))
        self.assertNotIn("fault_schedule", "".join(prompts))
        self.assertNotIn("export credentials", "".join(prompts))

    def test_all_resource_limits_terminate_without_hidden_calls(self):
        cases = [(Budgets(0, 4, 1000, 16), "tool_budget", 0, 0),
                 (Budgets(8, 0, 1000, 16), "model_budget", 1, 0),
                 (Budgets(8, 4, 25, 16), "token_budget", 1, 0)]
        for budgets, reason, tools, models in cases:
            result, _, _ = self.execute(budgets=budgets)
            self.assertEqual((result["stop_reason"], result["tool_calls"], result["model_calls"]), (reason, tools, models))
            self.assertTrue(result["abstained"])

    def test_transient_calls_count_even_when_all_fail(self):
        result, environment, _ = self.execute(faults={i: "transient" for i in range(1, 10)}, budgets=Budgets(3, 4, 1000, 16))
        self.assertEqual((result["tool_calls"], result["tool_failures"], result["model_calls"]), (3, 3, 0))
        self.assertEqual(environment.balance, 0)

    def test_negative_noninteger_budget_and_tokenizer_values_rejected(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                Budgets(8, value, 1000, 16)
        for value in (-1, True, 1.5, 0):
            result, _, _ = self.execute(count=lambda _, count=value: count)
            self.assertEqual((result["stop_reason"], result["model_calls"]), ("token_counter_failure", 0))

    def test_invalid_and_excess_usage_stops_without_budget_refund(self):
        for usage in ((-1, 2), (10, -1), (True, 2), (10, 1.5), (None, 2), (9, 2), (11, 2), (10, 17)):
            result, environment, events = self.execute(lambda req, u=usage: ModelReply(legal_reply(req).text, *u))
            self.assertEqual(result["stop_reason"], "provider_usage_contract")
            self.assertFalse(result["budget_guarantee_valid"])
            self.assertEqual(result["token_debits"], 26)
            self.assertEqual(environment.balance, 0)
            self.assertIn("raw_synthetic_model_output", events[-1])

    def test_wrong_provider_envelope_preserves_raw_string(self):
        raw = "malformed envelope with synthetic output"
        result, _, events = self.execute(lambda _: raw)
        self.assertEqual(result["stop_reason"], "provider_reply_contract")
        self.assertEqual(result["unknown_usage_calls"], 1)
        self.assertEqual(events[-1]["raw_synthetic_model_output"], raw)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("owned_model_runner", Path("scripts/run_model_recovery.py"))
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_frozen_receipts_exist_before_callback_and_refuse_overwrite(self):
        config = json.loads(Path("configs/model-policy-v1.json").read_text())
        config.update(execution_mode="unit-mock-validation", task_count_per_split=2, seeds=[11])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "config.json", root / "run"
            source.write_text(json.dumps(config))
            def provider(request):
                self.assertTrue((output / "initial_receipt.json").exists())
                self.assertTrue((output / "sources.json").exists())
                self.assertIn('"phase": "dispatch"', (output / "events.jsonl").read_text())
                return self.runner.mock_generate(request)
            report = self.runner.run_experiment(source, output, provider, self.runner.mock_token_count,
                provider_identity={"evidence_mode": "unit-mock-validation"})
            self.assertEqual(report["evidence_mode"], "unit-mock-validation")
            self.assertEqual(len(report["runs"]), 4)
            self.assertTrue((output / "completion_receipt.json").exists())
            self.assertTrue(all(row["model_calls"] == 0 for row in report["runs"] if row["policy"] == "step-retry"))
            with self.assertRaises(FileExistsError):
                self.runner.run_experiment(source, output, provider, self.runner.mock_token_count,
                    provider_identity={"evidence_mode": "unit-mock-validation"})

    def test_initial_receipt_preserved_if_execution_aborts(self):
        config = json.loads(Path("configs/model-policy-v1.json").read_text())
        config.update(execution_mode="unit-mock-validation", task_count_per_split=1, seeds=[11])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "config.json", root / "run"
            source.write_text(json.dumps(config))
            def abort(_):
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                self.runner.run_experiment(source, output, abort, self.runner.mock_token_count,
                    provider_identity={"evidence_mode": "unit-mock-validation"})
            self.assertEqual(json.loads((output / "initial_receipt.json").read_text())["status"], "started_not_complete")
            self.assertEqual(json.loads((output / "failure_receipt.json").read_text())["status"], "failed")
            failure = json.loads((output / "failure_receipt.json").read_text())
            self.assertIn("events.jsonl", failure["partial_artifact_hashes"])
            self.assertIn('"kind": "completed_case_metrics"', (output / "events.jsonl").read_text())
            self.assertFalse((output / "completion_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
