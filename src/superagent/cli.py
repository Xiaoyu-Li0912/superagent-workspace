import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--runtime", type=Path, default=Path("configs/runtime.yaml"))
    run.add_argument("--policy", type=Path, default=Path("configs/model_policy.json"))
    run.add_argument("--budget-rmb", type=float)
    run.add_argument("--steps", type=int)
    run.add_argument("--key-stdin", action="store_true", help="Read credential from private stdin, never command-line arguments")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--data-root", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--features", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == "run":
        if args.key_stdin:
            import os
            import sys
            os.environ["SUPERAGENT_API_KEY"] = sys.stdin.readline().strip()
        from .orchestrator import run_search, validate_runtime
        task = OmegaConf.to_container(OmegaConf.load(args.task), resolve=True)
        runtime = OmegaConf.to_container(OmegaConf.load(args.runtime), resolve=True)
        if args.budget_rmb is not None:
            runtime["budget_rmb"] = args.budget_rmb
        if args.steps is not None:
            runtime["steps"] = args.steps
        validate_runtime(runtime)
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        result = run_search(root, task, args.data_root.resolve(), args.run_dir, runtime, policy)
    elif args.command == "evaluate":
        from .orchestrator import final_evaluate
        result = final_evaluate(args.run_dir.resolve(), args.data_root.resolve())
    elif args.command == "predict":
        from .orchestrator import predict
        predict(args.run_dir.resolve(), args.features.resolve(), args.output.resolve())
        result = {"predictions": str(args.output / "predictions.csv")}
    else:
        from .budget import Ledger
        from .audit import write_json
        policy = json.loads((args.run_dir / "model_policy.json").read_text(encoding="utf-8"))
        runtime = json.loads((args.run_dir / "runtime.json").read_text(encoding="utf-8"))
        ledger = Ledger(args.run_dir / "cost.sqlite", runtime["budget_rmb"], policy["input_rmb_per_million"], policy["output_rmb_per_million"],
                        runtime["max_calls"], runtime.get("max_unknown_calls", 0))
        result = ledger.summary()
        write_json(args.run_dir / "cost_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
