import json
import logging
import math
import subprocess
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .artifacts import freeze, verify_frozen
from .audit import Audit, write_json, file_manifest
from .budget import Ledger, BudgetStop
from .contracts import CandidateResult
from .data import prepare_development, load_split, export_features
from .evaluator import evaluate_predictions
from .llm_gateway import Gateway
from .mlevolve_adapter import create_search
from .runner import Sandbox


def validate_runtime(runtime):
    if runtime.get("input_reservation_mode", "request_bytes") not in ("request_bytes", "full_context"):
        raise ValueError("input_reservation_mode must be request_bytes or full_context")
    for key in ("steps", "max_calls", "candidate_memory_mb", "candidate_cpus", "max_output_tokens"):
        if type(runtime[key]) is not int or runtime[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("time_limit_seconds", "candidate_timeout_seconds", "request_timeout_seconds", "budget_rmb"):
        value = runtime.get(key, 240) if key == "request_timeout_seconds" else runtime[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if runtime["budget_rmb"] > 20:
        raise ValueError("Per-task budget cannot exceed the course cap of RMB 20")
    for key in ("max_unknown_calls", "network_retry_attempts"):
        value = runtime.get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    for key in ("network_retry_backoff_seconds", "network_cooldown_seconds"):
        value = runtime.get(key, 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be non-negative and finite")


def run_search(root, task, data_root, run, runtime, policy):
    validate_runtime(runtime)
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    audit = Audit(run)
    write_json(run / "runtime.json", runtime)
    write_json(run / "task.json", task)
    write_json(run / "model_policy.json", policy)
    write_json(run / "source_hashes.json", file_manifest(root / "src"))
    write_json(run / "upstream_source_hashes.json", {
        name: digest for name, digest in file_manifest(root / "third_party/MLEvolve").items()
        if ".git" not in Path(name).parts and "__pycache__" not in Path(name).parts
        and not name.endswith(".pyc")
    })
    (run / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    audit.event("run_started", formal_cap_rmb=20, authorized_experiment_cap_rmb=runtime["budget_rmb"])
    sandbox = Sandbox(runtime["candidate_timeout_seconds"], runtime["candidate_memory_mb"], runtime["candidate_cpus"])
    profile = prepare_development(task, data_root, run)
    ledger = Ledger(run / "cost.sqlite", runtime["budget_rmb"], policy["input_rmb_per_million"], policy["output_rmb_per_million"],
                    runtime["max_calls"], runtime.get("max_unknown_calls", 0))
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(run / "mlevolve.log", encoding="utf-8"), logging.StreamHandler()])
    results = {}
    gateway = None
    try:
        gateway = Gateway(policy, ledger, audit, runtime["max_output_tokens"], runtime.get("request_timeout_seconds", 240),
                          runtime.get("network_retry_attempts", 0), runtime.get("network_retry_backoff_seconds", 0),
                          runtime.get("network_cooldown_seconds", 0),
                          runtime.get("input_reservation_mode", "request_bytes"))
        gateway.deadline = time.monotonic() + runtime["time_limit_seconds"]
        search = create_search(root / "third_party/MLEvolve", run, runtime, profile, task["description"], gateway, results, audit)
    except BaseException as error:
        cost = ledger.summary()
        reason = getattr(gateway, "stop_reason", None) or "initialization_error"
        write_json(run / "cost_summary.json", cost)
        write_json(run / "summary.json", {"status": "failed", "stop_reason": reason,
                   "candidates": 0, "valid_candidates": 0, "cost": cost})
        audit.event("initialization_failed", reason=reason, error_type=type(error).__name__)
        raise
    from engine.executor import ExecutionResult
    from engine.scheduler import run_search as schedule_search
    from config import save_run
    from llm import SearchStopped

    def execute(code, node_id, reset=True):
        start = time.monotonic()
        candidate = run / "candidates" / node_id
        candidate.mkdir(parents=True)
        (candidate / "solution.py").write_text(code, encoding="utf-8")
        audit.event("candidate_started", candidate_id=node_id)
        try:
            compile(code, "solution.py", "exec")
            sandbox.run(candidate / "solution.py", run / "data/train", candidate / "model", candidate / "train_output", "train", candidate / "train.log")
            manifest = file_manifest(candidate / "model")
            if not manifest:
                raise ValueError("Training did not save a model")
            sandbox.run(candidate / "solution.py", run / "data/val", candidate / "model", candidate / "val_output", "predict", candidate / "predict.log")
            labels = np.load(run / "private/validation_labels.npy", allow_pickle=False)
            metric = evaluate_predictions(candidate / "val_output/predictions.csv", labels, profile["classes"])
            shutil.copyfile(candidate / "val_output/predictions.csv",
                            search.cfg.workspace_dir / "submission" / f"submission_{node_id}.csv")
            result = CandidateResult(node_id, "valid", metric, time.monotonic() - start,
                                     f"Trusted validation accuracy={metric:.8f}; fresh-process reload passed.", str(candidate))
        except Exception as error:
            result = CandidateResult(node_id, "failed", None, time.monotonic() - start, f"{type(error).__name__}: {error}"[-18000:], str(candidate))
        results[node_id] = result
        write_json(candidate / "result.json", result.to_dict())
        audit.event("candidate_finished", **result.to_dict())
        return ExecutionResult([result.diagnostics], result.elapsed_seconds, None if result.status == "valid" else "CandidateFailure")

    stop = "step_limit"
    def checkpoint(agent):
        save_run(agent.cfg, agent.journal)
        write_json(run / "candidates.json", [r.to_dict() for r in list(results.values())])
        write_json(run / "decisions.json", [{"id": n.id, "parent": n.parent.id if n.parent else None,
                    "stage": n.stage, "plan": n.plan, "analysis": n.analysis,
                    "metric": n.metric.value if n.metric else None} for n in list(agent.journal.nodes)])
        write_json(run / "checkpoint.json", {"status": "searching", "journal_candidates": len(agent.journal) - 1,
                   "executed_candidates": len(results), "upstream_best": agent.best_node.id if agent.best_node else None,
                   "cost": ledger.summary()})
        audit.event("search_checkpoint", journal_candidates=len(agent.journal) - 1,
                    upstream_best=agent.best_node.id if agent.best_node else None)

    def should_stop():
        expired = time.monotonic() >= gateway.deadline
        if expired:
            if getattr(gateway, "stop_reason", None) is None:
                gateway.stop_reason = "time_limit"
            gateway.stopped.set()
        return expired or gateway.stopped.is_set()

    try:
        schedule_search(search, execute, checkpoint, should_stop)
        if should_stop():
            stop = getattr(gateway, "stop_reason", None) or "budget_or_uncertain_usage"
    except (BudgetStop, SearchStopped) as error:
        stop = (getattr(gateway, "stop_reason", None) or getattr(error, "reason", None)
                or getattr(error.__cause__, "reason", None) or "budget_or_uncertain_usage")
        if stop == "budget_or_uncertain_usage" and time.monotonic() >= gateway.deadline:
            stop = "time_limit"
    except Exception as error:
        stop = "search_error"
        audit.event("search_error", error_type=type(error).__name__, message=str(error))
    finally:
        write_json(run / "cost_summary.json", ledger.summary())
    # Respect upstream analysis/leakage checks and automatic best selection,
    # rather than independently overriding it with max(raw supervisor scores).
    valid = [n for n in search.journal.nodes if not n.is_buggy and n.is_valid is True
             and n.id in results and results[n.id].status == "valid"]
    if not valid:
        audit.event("run_failed", reason=stop, candidates=len(results))
        write_json(run / "summary.json", {"status": "failed", "stop_reason": stop, "candidates": len(results),
                   "valid_candidates": 0, "completed_steps": len(search.journal) - 1, "cost": ledger.summary()})
        raise RuntimeError("No valid reloadable autonomous candidate; inspect run audit")
    if search.best_node not in valid:
        raise RuntimeError("MLEvolve did not select a valid best node; refusing host-side reselection")
    best = results[search.best_node.id]
    freeze(Path(best.artifact_dir), run, best.to_dict())
    verify_frozen(run)
    summary = {"status": "frozen", "stop_reason": stop, "candidates": len(results), "valid_candidates": len(valid),
               "failed_candidates": sum(r.status != "valid" for r in results.values()),
               "completed_steps": len(search.journal) - 1,
               "best_candidate": best.candidate_id, "validation_accuracy": best.metric, "cost": ledger.summary()}
    write_json(run / "summary.json", summary)
    write_json(run / "checkpoint.json", summary)
    audit.event("run_frozen", candidate_id=best.candidate_id, stop_reason=stop)
    return summary


def predict(run, features, output):
    verify_frozen(run)
    runtime_path = run / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    sandbox = Sandbox(runtime.get("candidate_timeout_seconds", 300),
                      runtime.get("candidate_memory_mb", 4096), runtime.get("candidate_cpus", 2))
    model = run / "best/model"
    output.mkdir(parents=True, exist_ok=False)
    sandbox.run(run / "best/solution.py", features, model, output, "predict", output / "predict.log")
    verify_frozen(run)


def final_evaluate(run, data_root):
    verify_frozen(run)
    # A marker written BEFORE reading test prevents accidental repeated test feedback.
    with (run / "final_evaluation.lock").open("x", encoding="utf-8") as stream:
        stream.write("Final test access started; do not feed into search.\n")
    task = json.loads((run / "task.json").read_text(encoding="utf-8"))
    profile = json.loads((run / "profile.json").read_text(encoding="utf-8"))
    x, y = load_split(task, data_root, "test")
    export_features(x, run / "final_features", profile, task, data_root)
    predict(run, run / "final_features", run / "final_output")
    accuracy = evaluate_predictions(run / "final_output/predictions.csv", y, profile["classes"])
    result = {"test_accuracy": accuracy, "test_rows": len(y), "frozen_before_test": True}
    write_json(run / "final_result.json", result)
    Audit(run).event("final_evaluated", **result)
    return result
