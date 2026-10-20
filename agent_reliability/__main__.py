import argparse
from pathlib import Path
from .experiment import run
from .report import render


def main():
    parser = argparse.ArgumentParser(description="Synthetic replay harness; never calls a model")
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("run")
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.config, args.output)
        print("Deterministic synthetic replay complete:", len(result["runs"]), "runs")
    else:
        render(args.input, args.output)
        print("Report rendered from real JSON:", args.output)


if __name__ == "__main__":
    main()
