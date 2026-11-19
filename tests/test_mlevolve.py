"""Offline integration tests. Fakes replace paid transport/embeddings, not search policy."""
import sys
import json
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from superagent.audit import Audit
from superagent.budget import BudgetStop
from superagent.contracts import CandidateResult
from superagent.mlevolve_adapter import create_search, load_search_config, preflight

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/MLEvolve"
sys.path.insert(0, str(UPSTREAM))


class FakeMemory:
    def __init__(self, **kwargs):
        self.saved = []

    def save_node(self, node, parent):
        self.saved.append(node.id)


class OfflineGateway:
    def __init__(self):
        self.calls = []
        self.stopped = threading.Event()

    def query(self, system_message=None, func_spec=None, **kwargs):
        self.calls.append(("query", func_spec.name if func_spec else "clean"))
        if func_spec is None:
            return "Task: classification. Evaluation metric: accuracy, maximize."
        if func_spec.name == "determine_metric_direction":
            return {"lower_is_better": False, "reasoning": "accuracy"}
        if func_spec.name == "submit_code_review":
            return {"needs_revision": False, "reasoning": "Approved", "revised_code": None}
        if func_spec.name == "submit_review":
            return {"is_bug": False, "summary": "Measured run", "metric": 0.99,
                    "lower_is_better": False, "code_summary": "Persisted classifier"}
        raise AssertionError(func_spec.name)

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return "A concrete plan.\n```python\nprint('offline fixture')\n```"


class MLEvolveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        self.runtime = {"steps": 5, "time_limit_seconds": 60, "candidate_cpus": 2,
                        "candidate_timeout_seconds": 10, "seed": 42, "model": "DeepSeek-V4.1-Flash"}
        self.gateway = OfflineGateway()
        self.results = {}
        memory_module = types.ModuleType("agents.memory.global_memory")
        memory_module.GlobalMemoryLayer = FakeMemory
        self.memory_patch = patch.dict(sys.modules, {"agents.memory.global_memory": memory_module})
        self.memory_patch.start()
        self.addCleanup(self.memory_patch.stop)
        self.preflight_patch = patch("superagent.mlevolve_adapter.preflight")
        self.preflight_patch.start()
        self.addCleanup(self.preflight_patch.stop)

    def search(self):
        return create_search(UPSTREAM, self.run, self.runtime, {"classes": [0, 1]},
                             "Classification accuracy", self.gateway, self.results, Audit(self.run))

    def test_algorithm_defaults_are_not_overridden(self):
        cfg = load_search_config(UPSTREAM, self.run, self.runtime)
        original = OmegaConf.load(UPSTREAM / "config/config.yaml")
        actual_agent = OmegaConf.to_container(cfg.agent)
        expected_agent = OmegaConf.to_container(original.agent)
        for name in ("steps", "time_limit", "seed", "code", "feedback"):
            actual_agent.pop(name)
            expected_agent.pop(name)
        self.assertEqual(actual_agent, expected_agent)
        self.assertTrue(cfg.coldstart.use_coldstart)
        self.assertEqual(cfg.agent.initial_drafts, 3)
        self.assertEqual(cfg.agent.search.parallel_search_num, 3)

    def test_host_cannot_hide_algorithm_overrides(self):
        self.runtime["upstream_environment"] = {"agent.use_evolution": False}
        with self.assertRaisesRegex(ValueError, "Algorithm settings belong"):
            load_search_config(UPSTREAM, self.run, self.runtime)

    def test_missing_memory_fails_without_paid_calls(self):
        cfg = load_search_config(UPSTREAM, self.run, self.runtime)
        with patch("importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "No fallback"):
                preflight(cfg)
        self.assertEqual(self.gateway.calls, [])

    def test_actual_stepwise_draft_review_parse_memory_and_best(self):
        search = self.search()
        from engine.executor import ExecutionResult

        def execute(code, node_id, reset=True):
            self.results[node_id] = CandidateResult(node_id, "valid", 0.7, 1, "trusted accuracy=0.7", "fixture")
            path = search.cfg.workspace_dir / "submission" / f"submission_{node_id}.csv"
            path.write_text("row_id,prediction\n0,1\n")
            return ExecutionResult(["trusted accuracy=0.7"], 1, None)

        node = search.step(search.virtual_root, execute)
        self.assertEqual(len([c for c in self.gateway.calls if c[0] == "generate"]), 4)
        self.assertIn(("query", "submit_code_review"), self.gateway.calls)
        self.assertIn(("query", "submit_review"), self.gateway.calls)
        self.assertEqual(search.best_node.metric.value, 0.7)  # not the LLM's 0.99
        self.assertEqual(search.best_node.code_summary, "Persisted classifier")
        self.assertIn(search.best_node.id, search.global_memory.saved)
        self.assertTrue((search.cfg.workspace_dir / "best_solution/solution.py").exists())

    def test_stagnation_routes_to_evolution_and_fusion(self):
        search = self.search()
        from engine.search_node import SearchNode
        from utils.metric import MetricValue
        parent = SearchNode(code="x=1", stage="improve", parent=search.virtual_root)
        parent.branch_id, parent.is_buggy, parent.is_valid = 1, False, True
        parent.metric = MetricValue(0.7, maximize=True)
        search.search_start_time = time.time() - 40
        with patch("engine.agent_search.is_branch_stagnant", return_value=True), \
             patch("agents.evolution_agent.run", return_value=None) as evolution, \
             patch("agents.fusion_agent.run", return_value=None) as fusion:
            with patch("engine.agent_search.random.random", return_value=0.9):
                search._run_single_step(parent, lambda *args: None)
            evolution.assert_called_once()
            with patch("engine.agent_search.random.random", return_value=0.0):
                search._run_single_step(parent, lambda *args: None)
            fusion.assert_called_once()

    def test_review_keeps_search_replace_schema(self):
        self.search()
        from agents.code_review_agent import CODE_REVIEW_SPEC
        schema = CODE_REVIEW_SPEC.json_schema["properties"]["revised_code"]
        self.assertIn("SEARCH/REPLACE", schema["description"])
        self.assertIn("null", schema["type"])

    def test_native_diff_generation_applies_patch(self):
        search = self.search()
        from agents.coder.diff_coder.diff_generate import diff_generate_and_apply
        response = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"
        with patch.object(self.gateway, "generate", return_value=response):
            plan, code = diff_generate_and_apply(search, {"module": ["model"], "plan": {"model": "change x"}},
                                                "x = 1\n", "training data", "accuracy=0.7", "Improve")
        self.assertEqual(code.strip(), "x = 2")
        self.assertIn("change x", plan)

    def test_host_prompts_change_io_without_removing_agent_roles(self):
        search = self.search()
        from agents.coder.stepwise_coder import create_default_step_agents
        from agents.prompts.validation_template_prompts import get_code_review_guidelines
        native = create_default_step_agents()
        hosted = create_default_step_agents(search.integration)
        self.assertEqual([a.name for a in native], [a.name for a in hosted])
        self.assertIn("./input", native[0].description)
        self.assertIn("/data/features.csv", hosted[0].guidelines[0])
        guidelines = "\n".join(get_code_review_guidelines(search.integration))
        self.assertNotIn("9 hours available", guidelines)
        self.assertNotIn("ALL EXIST", guidelines)
        self.assertIn("P0.1 Data Leakage", guidelines)
        self.assertIn("train --data /data", guidelines)

    def test_native_aggregation_time_window_is_not_shortened(self):
        search = self.search()
        from engine.conditions import should_trigger_branch_fusion
        search.search_start_time = time.time() - 7200
        self.assertEqual(search.scfg.fusion_min_time_hours, 6)
        self.assertFalse(should_trigger_branch_fusion(search))

    def test_enabled_memory_initialization_failure_is_fatal(self):
        with patch.object(FakeMemory, "__init__", side_effect=OSError("model unavailable")):
            with self.assertRaisesRegex(RuntimeError, "refusing silent ablation"):
                self.search()
        self.assertEqual(self.gateway.calls, [])

    def test_budget_stop_bypasses_agent_retries(self):
        self.search()
        from llm import generate, SearchStopped
        with patch.object(self.gateway, "generate", side_effect=BudgetStop("cap")) as call:
            with self.assertRaises(SearchStopped):
                generate("x", cfg=load_search_config(UPSTREAM, self.run, self.runtime))
            call.assert_called_once()

    def test_transport_keeps_native_deepseek_reasoning_profile(self):
        search = self.search()
        from llm import generate
        generate("x", cfg=search.cfg)
        options = self.gateway.calls[-1][1]["provider_options"]
        self.assertEqual(options["thinking"], {"type": "enabled"})
        self.assertEqual(options["reasoning_effort"], "high")

    def test_two_phase_parallel_scheduler(self):
        from engine.scheduler import run_search
        events, lock = [], threading.Lock()
        active = 0
        peak = 0
        journal = [object()]

        def deferred(node, execute):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                events.append("execute")
            time.sleep(0.03)
            with lock:
                active -= 1
                journal.append(node)
            return node

        def step(parent, execute, execute_immediately=True):
            if not execute_immediately:
                events.append("draft")
                return types.SimpleNamespace(pending_execution=True)
            return deferred(parent, execute)

        agent = types.SimpleNamespace(acfg=types.SimpleNamespace(steps=6, initial_drafts=3),
                                      scfg=types.SimpleNamespace(parallel_search_num=3),
                                      journal=journal,
                                      step=step, execute_deferred_node=deferred)
        self.assertEqual(run_search(agent, None, lambda a: None, stagger_seconds=0), 6)
        self.assertEqual(events[:3], ["draft"] * 3)
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, 3)

    def test_scheduler_retries_failed_task_and_counts_journal_nodes(self):
        from engine.scheduler import run_search
        journal, calls = [object()], []

        def step(parent, execute, execute_immediately=True):
            calls.append(parent)
            if len(calls) == 1:
                raise ValueError("ordinary upstream failure")
            node = object()
            journal.append(node)
            return node

        agent = types.SimpleNamespace(acfg=types.SimpleNamespace(steps=2, initial_drafts=0),
                                      scfg=types.SimpleNamespace(parallel_search_num=1),
                                      journal=journal, step=step)
        with self.assertLogs("MLEvolve", level="ERROR"):
            self.assertEqual(run_search(agent, None, lambda a: None, stagger_seconds=0), 2)
        self.assertEqual(len(calls), 3)
        self.assertIsNone(calls[1])

    def test_scheduler_drains_running_workers_before_final_checkpoint(self):
        from engine.scheduler import run_search
        from llm import SearchStopped
        started, released = threading.Event(), threading.Event()
        checkpoints, journal = [], [object()]

        def step(parent, execute, execute_immediately=True):
            if not started.is_set():
                started.set()
                released.wait(timeout=5)
                journal.append(object())
                return object()
            released.set()
            raise SearchStopped("budget")

        agent = types.SimpleNamespace(acfg=types.SimpleNamespace(steps=2, initial_drafts=0),
                                      scfg=types.SimpleNamespace(parallel_search_num=2),
                                      journal=journal, step=step)
        with self.assertRaises(SearchStopped):
            run_search(agent, None, lambda a: checkpoints.append(len(journal)), stagger_seconds=0)
        self.assertEqual(checkpoints[-1], 2)

    @unittest.skipUnless(sys.platform == "linux", "Requires Linux bubblewrap")
    def test_synthetic_pipeline_uses_native_parallel_search_and_freezes(self):
        import pandas as pd
        from engine.scheduler import run_search as native_scheduler
        from superagent.orchestrator import run_search
        from superagent.artifacts import verify_frozen
        data = self.run / "synthetic"
        data.mkdir()
        for split in ("train", "val"):
            pd.DataFrame({"feature": [0, 1, 2, 3], "label": [0, 1, 0, 1]}).to_csv(data / f"{split}.csv", index=False)
        task = {"format": "csv", "modality": "tabular", "feature_columns": ["feature"], "label_column": "label",
                "train_features": "train.csv", "val_features": "val.csv", "test_features": "FORBIDDEN.csv",
                "description": "Binary classification; accuracy"}
        runtime = {**self.runtime, "steps": 3, "time_limit_seconds": 120, "max_calls": 150,
                   "candidate_timeout_seconds": 30, "candidate_memory_mb": 4096, "budget_rmb": 10,
                   "max_output_tokens": 16384, "request_timeout_seconds": 900}
        policy = {"input_rmb_per_million": 1, "output_rmb_per_million": 4}
        code = '''import argparse, json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('mode'); p.add_argument('--data'); p.add_argument('--model'); p.add_argument('--output'); a=p.parse_args()
if a.mode=='train':
    y=pd.read_csv(Path(a.data)/'labels.csv'); (Path(a.model)/'model.json').write_text(json.dumps(int(y.label.mode()[0])))
else:
    label=json.loads((Path(a.model)/'model.json').read_text())
    x=pd.read_csv(Path(a.data)/'features.csv')
    pd.DataFrame({'row_id':range(len(x)), 'prediction':[label]*len(x)}).to_csv(a.output,index=False)
'''
        def schedule(*args, **kwargs):
            return native_scheduler(*args, **kwargs, stagger_seconds=0)

        run = self.run / "pipeline"
        with patch("superagent.orchestrator.Gateway", return_value=self.gateway) as gateway_factory, \
             patch.object(self.gateway, "generate", return_value="Fixture plan.\n```python\n" + code + "\n```"), \
             patch("engine.scheduler.run_search", side_effect=schedule):
            summary = run_search(ROOT, task, data, run, runtime, policy)
            self.assertEqual(gateway_factory.call_args.args[3:], (16384, 900, 0, 0, 0, "request_bytes"))
        self.assertEqual(summary["status"], "frozen")
        self.assertEqual(summary["candidates"], 3)
        self.assertEqual(summary["validation_accuracy"], 0.5)
        self.assertEqual(json.loads((run / "checkpoint.json").read_text()), summary)
        verify_frozen(run)
        self.assertFalse((run / "final_evaluation.lock").exists())
        integration = json.loads((run / "integration.json").read_text())
        self.assertTrue(integration["memory_initialized"])
        self.assertFalse(integration["coldstart_guidance_found"])


if __name__ == "__main__":
    unittest.main()
