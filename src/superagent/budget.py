"""SQLite reservations survive crashes and serialize concurrent budget checks."""
import json
import sqlite3
import time
import uuid
from decimal import Decimal, ROUND_CEILING
from pathlib import Path


class BudgetStop(RuntimeError):
    stop_search = True

    def __init__(self, message, reason="budget_or_uncertain_usage"):
        super().__init__(message)
        self.reason = reason


def cost_fen_units(tokens_in, tokens_out, input_rate, output_rate):
    # Integer nanoyuan avoids floating point comparisons at the hard cap.
    value = (Decimal(tokens_in) * Decimal(input_rate) + Decimal(tokens_out) * Decimal(output_rate)) * 1000
    return int(value.to_integral_value(rounding=ROUND_CEILING))


class Ledger:
    def __init__(self, path: Path, cap_rmb, input_rate, output_rate, max_calls=24,
                 max_unknown_calls=0):
        self.path = path
        self.cap = int(Decimal(str(cap_rmb)) * 1_000_000_000)
        self.input_rate, self.output_rate = str(input_rate), str(output_rate)
        self.max_calls = max_calls
        if type(max_unknown_calls) is not int or max_unknown_calls < 0:
            raise ValueError("max_unknown_calls must be a non-negative integer")
        self.max_unknown_calls = max_unknown_calls
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, created REAL, state TEXT, reserve INTEGER, charged INTEGER, input_tokens INTEGER, output_tokens INTEGER, usage TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY, cap INTEGER, input_rate TEXT, output_rate TEXT, max_calls INTEGER)")
            expected = (self.cap, self.input_rate, self.output_rate, max_calls)
            row = db.execute("SELECT cap,input_rate,output_rate,max_calls FROM policy WHERE id=1").fetchone()
            if row is None:
                db.execute("INSERT INTO policy VALUES (1,?,?,?,?)", expected)
            elif row != expected:
                raise ValueError("Cannot change the policy of an existing expense ledger")
            recovery = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='recovery_policy'").fetchone()
            if recovery:
                saved = db.execute("SELECT max_unknown_calls FROM recovery_policy WHERE id=1").fetchone()
                if saved != (max_unknown_calls,):
                    raise ValueError("Cannot change the recovery policy of an existing expense ledger")
            elif row is None:
                db.execute("CREATE TABLE recovery_policy (id INTEGER PRIMARY KEY, max_unknown_calls INTEGER)")
                db.execute("INSERT INTO recovery_policy VALUES (1,?)", (max_unknown_calls,))
            elif max_unknown_calls != 0:
                raise ValueError("Legacy ledger has a strict recovery policy; use a new run")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30, isolation_level="IMMEDIATE")

    def reserve(self, input_upper, output_upper):
        amount = cost_fen_units(input_upper, output_upper, self.input_rate, self.output_rate)
        call_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_dispatch(db, amount)
            db.execute("INSERT INTO calls VALUES (?,?, 'reserved',?,?,NULL,NULL,NULL)", (call_id, time.time(), amount, amount))
        return call_id

    def check_dispatch(self, input_upper, output_upper):
        amount = cost_fen_units(input_upper, output_upper, self.input_rate, self.output_rate)
        with self.connect() as db:
            self._check_dispatch(db, amount)

    def _check_dispatch(self, db, amount):
        spent, count, unknown = db.execute("SELECT COALESCE(SUM(charged),0),COUNT(*),COALESCE(SUM(state='unknown'),0) FROM calls").fetchone()
        if unknown >= max(1, self.max_unknown_calls):
            raise BudgetStop("Unknown usage limit reached; reservations retained", "unknown_usage_limit")
        if count >= self.max_calls:
            raise BudgetStop("Call limit reached", "call_limit")
        if spent + amount > self.cap:
            raise BudgetStop("Reservation would exceed budget", "budget_limit")

    def settle(self, call_id, input_tokens, output_tokens, usage):
        if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
            self.unknown(call_id)
            raise BudgetStop("Missing or invalid token usage; reservation retained", "invalid_usage")
        amount = cost_fen_units(input_tokens, output_tokens, self.input_rate, self.output_rate)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reserve,state FROM calls WHERE id=?", (call_id,)).fetchone()
            if not row or row[1] != "reserved":
                raise ValueError("Invalid settlement")
            db.execute("UPDATE calls SET state=?,charged=?,input_tokens=?,output_tokens=?,usage=? WHERE id=?", ("settled" if amount <= row[0] else "unknown", max(amount, row[0]) if amount > row[0] else amount, input_tokens, output_tokens, json.dumps(usage), call_id))
        if amount > row[0]:
            raise BudgetStop("Provider exceeded reserved usage; halt and reconcile", "reservation_exceeded")

    def unknown(self, call_id):
        with self.connect() as db:
            db.execute("UPDATE calls SET state='unknown' WHERE id=? AND state='reserved'", (call_id,))

    def summary(self):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            calls = [dict(row) for row in db.execute("SELECT * FROM calls ORDER BY created")]
        return {"cap_rmb": self.cap / 1e9, "charged_or_reserved_rmb": sum(c["charged"] for c in calls) / 1e9,
                "settled_rmb": sum(c["charged"] for c in calls if c["state"] == "settled") / 1e9,
                "unknown_reserved_rmb": sum(c["charged"] for c in calls if c["state"] == "unknown") / 1e9,
                "inflight_reserved_rmb": sum(c["charged"] for c in calls if c["state"] == "reserved") / 1e9,
                "max_unknown_calls": self.max_unknown_calls,
                "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
                "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
                "unresolved_calls": sum(c["state"] != "settled" for c in calls), "calls": calls}
