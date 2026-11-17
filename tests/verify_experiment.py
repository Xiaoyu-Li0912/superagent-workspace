"""Read-only verification of raw responses, SQLite ledgers and frozen run evidence."""
import hashlib
import json
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    runs = root / "runs"
    attempts = []
    input_total = output_total = reasoning_total = 0
    money = Decimal(0)
    for run in sorted(runs.glob("coupon-autonomous-*")):
        ledger_path = run / "cost.sqlite"
        if not ledger_path.exists():
            assert not list((run / "calls").glob("*.request.json"))
            attempts.append({"run": run.name, "calls": 0, "rmb": "0"})
            continue
        policy = read(run / "model_policy.json")
        rate_in = Decimal(policy["input_rmb_per_million"])
        rate_out = Decimal(policy["output_rmb_per_million"])
        with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            calls = list(db.execute("SELECT * FROM calls ORDER BY created"))
        assert len(calls) == len(list((run / "calls").glob("*.request.json")))
        assert len(calls) == len(list((run / "calls").glob("*.response.json")))
        subtotal = Decimal(0)
        run_input = run_output = 0
        for call in calls:
            assert call["state"] == "settled", "Unresolved billed dispatch"
            request = read(run / "calls" / f'{call["id"]}.request.json')
            response = read(run / "calls" / f'{call["id"]}.response.json')
            usage = response["usage"]
            tin, tout = usage["prompt_tokens"], usage["completion_tokens"]
            assert type(tin) is int and type(tout) is int and min(tin, tout) >= 0
            assert usage["total_tokens"] == tin + tout
            assert json.loads(call["usage"]) == usage
            assert (tin, tout) == (call["input_tokens"], call["output_tokens"])
            assert tout <= request["max_tokens"] and tin <= policy["context_tokens_upper_bound"]
            assert request["model"] == policy["model"] == response["model"]
            cost = (tin * rate_in + tout * rate_out) / 1_000_000
            assert cost * 1_000_000_000 == call["charged"]
            subtotal += cost
            run_input += tin
            run_output += tout
            reasoning_total += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        assert subtotal <= Decimal(str(read(run / "runtime.json")["budget_rmb"]))
        attempts.append({"run": run.name, "calls": len(calls), "input_tokens": run_input,
                         "output_tokens": run_output, "rmb": str(subtotal)})
        input_total += run_input
        output_total += run_output
        money += subtotal
    assert money <= 30
    run = runs / "coupon-autonomous-03"
    for relative, expected in read(run / "source_hashes.json").items():
        assert digest(root / "src" / relative) == expected, "Runtime source changed"
    frozen = read(run / "freeze.json")
    for relative, expected in frozen["files"].items():
        assert digest(run / "best" / relative) == expected
    assert set(frozen["files"]) == {str(p.relative_to(run / "best")) for p in (run / "best").rglob("*") if p.is_file()}
    candidates = read(run / "candidates.json")
    valid = [c for c in candidates if c["status"] == "valid"]
    assert max(valid, key=lambda c: c["metric"])["candidate_id"] == frozen["candidate"]["candidate_id"]
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    freeze_time = next(e["time_unix"] for e in events if e["event"] == "run_frozen")
    final_events = [e for e in events if e["event"] == "final_evaluated"]
    assert len(final_events) == 1 and final_events[0]["time_unix"] > freeze_time
    assert (run / "final_evaluation.lock").is_file()
    final = read(run / "final_result.json")
    assert final["frozen_before_test"]
    import csv
    with (run / "final_output/predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == final["test_rows"]
    assert [int(row["row_id"]) for row in predictions] == list(range(final["test_rows"]))
    return {"verified": True, "attempts": attempts, "input_tokens": input_total,
            "output_tokens": output_total, "reasoning_tokens_included_in_output": reasoning_total,
            "total_tokens": input_total + output_total, "total_rmb": str(money), "cap_rmb": 30,
            "unresolved_calls": 0, "candidates": len(candidates), "valid_candidates": len(valid),
            "best_candidate": frozen["candidate"]["candidate_id"],
            "validation_accuracy": frozen["candidate"]["metric"], "final_result": final,
            "pricing_evidence_sha256": {p.name: digest(p) for p in (root / "configs/pricing-evidence").iterdir()},
            "limits": ["Provider exact model build not independently attested", "Test labels not reopened by this audit",
                       "One task experiment, not final course submission or multimodal performance proof"]}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = verify(root)
    if "--save" in sys.argv:
        from superagent.audit import write_json
        write_json(root / "runs/experiment-audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
