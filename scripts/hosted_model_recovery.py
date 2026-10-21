"""Explicit opt-in reviewed free-hosted model-policy run. Never run on laptop."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_reliability.qwen_provider import hosted_location, load_provider
from run_model_recovery import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", required=True, choices=["free-colab", "free-kaggle"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hosted_location(args.location)  # Before network, model imports or local filesystem mutation.
    if args.output.exists():
        raise FileExistsError("Preserve existing run; select a unique output directory")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # This process only; public checkpoint.
    setup = args.output.with_name(args.output.name + "-setup.json")
    setup.parent.mkdir(parents=True, exist_ok=True)
    with setup.open("x", encoding="utf-8") as stream:
        json.dump({"status": "hosted_load_requested_not_inference_complete", "location": args.location,
                   "time": datetime.now(timezone.utc).isoformat(), "output": str(args.output)}, stream, indent=2)
    try:
        provider, identity = load_provider(args.location, args.output / "provider-trace.jsonl")
        report = run_experiment(args.config, args.output, provider.generate, provider.count_prompt_tokens,
            provider_identity=identity,
            provider_sources=[Path(__file__), ROOT / "agent_reliability/qwen_provider.py"])
        print(json.dumps({"status": "run_complete", "runs": len(report["runs"]),
                          "evidence_mode": report["evidence_mode"], "output": str(args.output)}))
    except BaseException as error:
        with setup.with_name(setup.stem + "-failure.json").open("x", encoding="utf-8") as stream:
            json.dump({"status": "failed_or_interrupted", "exception_type": type(error).__name__}, stream)
        raise


if __name__ == "__main__":
    main()
