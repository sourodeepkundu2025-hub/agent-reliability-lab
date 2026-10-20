"""Opt-in hosted GPU schema pilot; not the multi-step recovery experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import uuid

# Direct script invocation must find the sibling package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_reliability.contracts import parse_action, ContractError

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def cases():
    return [{"task": {"task_id": f"schema-{i}", "increments": [i + 2, i + 5, i + 1]},
             "next_step": i % 3} for i in range(8)]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_output(raw, case):
    try:
        parse_action(raw, case["task"], case["next_step"], case["next_step"])
        return True, None
    except (ContractError, IndexError) as exc:
        return False, str(exc)


def new_run_directory(path):
    # No implicit resume/merging across ephemeral GPUs. Existing evidence survives.
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_receipt(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "gpu-pilot" / (stamp + "-" + uuid.uuid4().hex[:8])))
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU allocated. Select a hosted GPU; CPU execution is not GPU evidence.")
    output = new_run_directory(args.output)
    config = {"model": MODEL, "revision": REVISION, "seed": 11, "max_new_tokens": 128,
              "do_sample": False, "enable_thinking": False, "cases": cases(),
              "purpose": "synthetic action-schema pilot, not a recovery or reasoning benchmark"}
    manifest = {"config": config, "config_sha256": digest(config),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "contract_sha256": hashlib.sha256((ROOT / "agent_reliability/contracts.py").read_bytes()).hexdigest(),
                "python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "device": "cuda:0", "dtype": "float16", "attention_backend": "eager",
                "status": "initializing", "completed": 0, "valid": 0,
                "limit": "Eight synthetic schema cases, greedy decoding; no training, held-out recovery, benchmark or publication claim."}
    # Keep the hashed initial identity immutable while the progress receipt changes.
    identity_bytes = json.dumps(manifest, sort_keys=True).encode()
    (output / "run_manifest.json").write_bytes(identity_bytes)
    manifest["run_manifest_sha256"] = hashlib.sha256(identity_bytes).hexdigest()
    receipt = output / "receipt.json"
    save_receipt(receipt, manifest)
    try:
        torch.manual_seed(11)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False, token=False)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            trust_remote_code=False, token=False, use_safetensors=True,
            torch_dtype=torch.float16, attn_implementation="eager").to("cuda:0")
        model.eval()
        if any(p.device.type != "cuda" or p.device.index != 0 for p in model.parameters()):
            raise RuntimeError("Model is not entirely on the recorded CUDA device.")
        manifest["status"] = "running"
        save_receipt(receipt, manifest)
        for case_id, case in enumerate(cases()):
            messages = [
                {"role": "system", "content": "Return one JSON object only with fields step, amount, version and integer values. amount is increments[step]; version equals the requested step. Do not call tools or write files."},
                {"role": "user", "content": json.dumps(case)}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter() - started
            raw = tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            valid, error = validate_output(raw, case)
            row = {"case_id": case_id, "run_manifest_sha256": manifest["run_manifest_sha256"],
                   "config_sha256": manifest["config_sha256"], "messages": messages,
                   "raw_output": raw, "valid_action": valid, "error": error,
                   "input_tokens": inputs["input_ids"].shape[-1],
                   "output_tokens": generated.shape[-1] - inputs["input_ids"].shape[-1],
                   "generation_seconds": elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                   "evidence_class": "actual_model_run"}
            with (output / "traces.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            manifest["completed"] += 1
            manifest["valid"] += int(valid)
            save_receipt(receipt, manifest)
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "interrupted_or_failed"
        raise
    finally:
        traces = output / "traces.jsonl"
        if traces.exists():
            manifest["traces_sha256"] = hashlib.sha256(traces.read_bytes()).hexdigest()
        save_receipt(receipt, manifest)
        print("Evidence directory:", output)
    print(json.dumps({k: manifest[k] for k in ["gpu", "completed", "valid", "limit"]}, indent=2))


if __name__ == "__main__":
    main()
