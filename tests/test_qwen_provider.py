"""Offline provider-boundary tests. No torch/transformers/network/model import."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from agent_reliability.qwen_provider import QwenProvider, hosted_location, load_provider


class Tokenizer:
    def __init__(self):
        self.calls = []
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return json.dumps(messages)
    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": list(range(len(text)))}


class ProviderTests(unittest.TestCase):
    def test_template_and_counter_are_exact_non_thinking_and_bounded(self):
        tokenizer = Tokenizer()
        provider = QwenProvider(tokenizer, None, None, "not-used")
        raw = '{"step":0}'
        self.assertEqual(provider.count_prompt_tokens(raw), len(provider.render(raw)))
        messages, kwargs = tokenizer.calls[0]
        self.assertEqual(messages[1]["content"], raw)
        self.assertEqual(kwargs, {"tokenize":False,"add_generation_prompt":True,"enable_thinking":False})
        with self.assertRaises(ValueError):
            provider.count_prompt_tokens("x" * 5000)

    def test_local_host_fails_before_model_imports_or_downloads(self):
        with mock.patch("agent_reliability.qwen_provider.Path.is_dir", return_value=False):
            for location in ["free-kaggle", "free-colab", "local", "paid"]:
                with self.subTest(location=location), self.assertRaises(RuntimeError):
                    load_provider(location, "never-created")

    def test_host_marker_is_required_not_only_directory(self):
        with mock.patch("agent_reliability.qwen_provider.Path.is_dir", return_value=True), mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                hosted_location("free-kaggle")
        with mock.patch("agent_reliability.qwen_provider.Path.is_dir", return_value=True), mock.patch.dict("os.environ", {"KAGGLE_KERNEL_RUN_TYPE":"Batch"}, clear=True):
            self.assertEqual(hosted_location("free-kaggle"), "free-kaggle")

    def test_invalid_cap_rejected_before_tokenization_or_gpu_access(self):
        provider = QwenProvider(None, None, None, "not-used")
        for cap in [0, -1, True, 129, 1.5]:
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                provider.generate(type("Request", (), {"max_output_tokens":cap})())

    def test_trace_retains_raw_synthetic_data_only_when_explicitly_called(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            provider = QwenProvider(None, None, None, path)
            self.assertFalse(path.exists())
            event = {"phase":"result", "raw_synthetic_model_output":"not JSON"}
            provider.write(event)
            self.assertEqual(json.loads(path.read_text()), event)


if __name__ == "__main__":
    unittest.main()
