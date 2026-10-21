"""Owned fixed-checkpoint adapter. Importing this module does not load a model."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from .model_policy import ModelReply

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SYSTEM = "Return exactly one JSON object with integer fields step, amount, version. Follow the public task instruction. No explanations, markdown or tools."


def hosted_location(location):
    """Fail closed before imports/downloads on a laptop or unselected host."""
    if location == "free-kaggle" and Path("/kaggle/working").is_dir() and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return location
    if location == "free-colab" and Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG"):
        return location
    raise RuntimeError("Reviewed free Colab/Kaggle runtime required; no local inference fallback")


class QwenProvider:
    def __init__(self, tokenizer, model, torch, trace_path):
        self.tokenizer, self.model, self.torch = tokenizer, model, torch
        self.trace_path = Path(trace_path)
        self.attempts = 0

    def render(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def count_prompt_tokens(self, prompt):
        tokens = self.tokenizer(self.render(prompt), add_special_tokens=False)["input_ids"]
        if not 0 < len(tokens) <= 4096:
            raise ValueError("Input exceeds the frozen 4096-token prompt cap")
        return len(tokens)

    def write(self, event):
        # Parent runner creates this unique output directory before any callback.
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()

    def generate(self, request):
        if type(request.max_output_tokens) is not int or not 1 <= request.max_output_tokens <= 128:
            raise ValueError("Output token cap exceeds reviewed provider limit")
        rendered = self.render(request.prompt)
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to("cuda:0")
        ninput = int(inputs["input_ids"].shape[-1])
        if ninput != self.count_prompt_tokens(request.prompt):
            raise RuntimeError("Dispatch tokenization differs from budget tokenizer")
        self.attempts += 1
        event = {"provider_attempt": self.attempts, "phase": "dispatch", "input_tokens": ninput,
                 "rendered_prompt": rendered, "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                 "input_ids_sha256": hashlib.sha256(json.dumps(inputs["input_ids"][0].tolist()).encode()).hexdigest(),
                 "max_new_tokens": request.max_output_tokens, "seed": request.seed,
                 "do_sample": False, "enable_thinking": False}
        self.write(event)  # Persist the actual model input before inference.
        torch = self.torch
        torch.manual_seed(request.seed)
        torch.cuda.manual_seed_all(request.seed)
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=request.max_output_tokens,
                                                do_sample=False, num_beams=1, use_cache=True)
            torch.cuda.synchronize(0)
            output = generated[0, ninput:]
            noutput = int(output.shape[-1])
            if noutput > request.max_output_tokens:
                raise RuntimeError("Provider exceeded generation token cap")
            raw = self.tokenizer.decode(output, skip_special_tokens=True)
            self.write({**event, "phase": "result", "outcome": "returned", "output_tokens": noutput,
                        "raw_synthetic_model_output": raw, "generation_seconds": time.perf_counter() - started,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)})
            return ModelReply(raw, ninput, noutput, True)
        except BaseException as error:
            self.write({**event, "phase": "result", "outcome": "failed", "exception_type": type(error).__name__})
            raise


def load_provider(location, trace_path):
    hosted_location(location)
    if importlib.metadata.version("transformers") != "4.57.1" or importlib.metadata.version("safetensors") != "0.6.2":
        raise RuntimeError("Use the frozen reviewed transformers/safetensors versions")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("No hosted CUDA allocation; no CPU fallback/model download")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False, token=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False,
        token=False, use_safetensors=True, torch_dtype=torch.float16, attn_implementation="eager").to("cuda:0")
    model.eval()
    if any(p.device.type != "cuda" or p.device.index != 0 for p in model.parameters()):
        raise RuntimeError("Model parameters differ from recorded CUDA:0 placement")
    provider = QwenProvider(tokenizer, model, torch, trace_path)
    identity = {"evidence_mode": "real-free-hosted-model", "execution_location": location,
                "model_id": MODEL, "checkpoint_revision": REVISION, "tokenizer_revision": REVISION,
                "tokenizer_class": type(tokenizer).__name__,
                "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "system_prompt": SYSTEM, "hardware": {"gpu": torch.cuda.get_device_name(0),
                    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "device": "cuda:0", "dtype": "float16", "attention_backend": "eager"},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "transformers": importlib.metadata.version("transformers"),
                    "safetensors": importlib.metadata.version("safetensors")},
                "sampling_parameters": {"do_sample": False, "num_beams": 1, "use_cache": True,
                    "enable_thinking": False, "max_input_tokens": 4096},
                "generation_config": model.generation_config.to_dict(),
                "scope": "Synthetic public ledger actions only; deterministic greedy decoding is an engineering pilot, not the model-card recommended sampling benchmark."}
    return provider, identity
