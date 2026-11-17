"""Course I/O integration; search policy and defaults belong to third_party/MLEvolve."""
import importlib.util
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from .audit import write_json
from .contracts import CANDIDATE_CONTRACT


class CourseIntegration:
    def __init__(self, profile, results, runtime):
        self.profile, self.results, self.runtime = profile, results, runtime

    def instructions(self):
        return CANDIDATE_CONTRACT + (
            f"\nActual execution limits: {self.runtime['candidate_cpus']} CPU threads, "
            f"{self.runtime.get('candidate_memory_mb', 4096)} MB address space, "
            f"{self.runtime['candidate_timeout_seconds']} seconds per train/predict process."
        )

    def environment(self):
        packages = [name for name in ("numpy", "pandas", "sklearn", "catboost", "lightgbm", "PIL", "torch", "transformers")
                    if importlib.util.find_spec(name) is not None]
        return {"Installed Packages": "CPU environment; " + ", ".join(packages) + ". "
                "No network or pretrained candidate cache. Consult the execution contract; "
                "do not assume the upstream benchmark's GPU/packages are available."}

    def data_preview(self):
        return json.dumps(self.profile, ensure_ascii=False)

    def validate(self, node):
        result = self.results[node.id]
        node.is_valid = result.status == "valid"
        if not node.is_valid:
            node.is_buggy = True
        return node.is_valid

    def authoritative_review(self, node, response):
        result = self.results[node.id]
        response = dict(response)
        # Preserve LLM analysis/code summary and bug judgement; only the measured
        # score comes from the supervisor, never from generated stdout.
        response["metric"] = result.metric
        response["summary"] = response.get("summary", "") + "\n" + result.diagnostics
        if result.status != "valid":
            response["is_bug"] = True
        return response


def load_search_config(upstream, run, runtime):
    cfg = OmegaConf.load(upstream / "config/config.yaml")
    overrides = runtime.get("upstream_environment", {})
    allowed = {"agent.memory_embedding_device", "agent.memory_embedding_model_path",
               "agent.search.num_gpus", "pretrain_model_dir", "torch_hub_dir"}
    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError(f"Algorithm settings belong in third_party/MLEvolve/config/config.yaml: {key}")
        OmegaConf.update(cfg, key, value)
    cfg._set_flag("allow_objects", True)
    cfg.workspace_dir = run / "mlevolve"
    cfg.log_dir = run / "mlevolve"
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(exist_ok=True)
    cfg.agent.steps = runtime["steps"]
    cfg.agent.time_limit = runtime["time_limit_seconds"]
    cfg.agent.seed = runtime["seed"]
    cfg.cpu_number = runtime["candidate_cpus"] * cfg.agent.search.parallel_search_num
    cfg.exec.timeout = runtime["candidate_timeout_seconds"]
    cfg.use_grading_server = False
    cfg.exp_name = "course_classification_task"
    cfg.exp_id = "course_classification"
    for key in ("task_json_path", "model_json_path"):
        path = Path(cfg.coldstart[key])
        cfg.coldstart[key] = str(path if path.is_absolute() else upstream / path)
    for stage in (cfg.agent.code, cfg.agent.feedback):
        stage.model = runtime["model"]
        stage.api_key = ""
        stage.base_url = ""
    return cfg


def preflight(cfg):
    if cfg.agent.use_global_memory:
        required = ("torch", "sentence_transformers", "faiss", "rank_bm25")
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError("MLEvolve memory is enabled; install its dependencies before running: "
                               + ", ".join(missing) + ". No fallback to memory-disabled search.")


def create_search(upstream: Path, run: Path, runtime, profile, description, gateway, results, audit):
    upstream = upstream.resolve()
    sys.path.insert(0, str(upstream))
    from llm import set_transport
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from engine.coldstart import build_guidance_description
    from utils.seed import set_global_seed

    cfg = load_search_config(upstream, run, runtime)
    preflight(cfg)
    if cfg.coldstart.use_coldstart:
        cfg.coldstart.description = build_guidance_description(cfg)
    OmegaConf.save(cfg, run / "mlevolve-effective.yaml")
    set_transport(gateway)
    set_global_seed(runtime["seed"])
    search = AgentSearch(description, cfg, Journal(), integration=CourseIntegration(profile, results, runtime))
    write_json(run / "integration.json", {
        "algorithm_config": "third_party/MLEvolve/config/config.yaml",
        "effective_config": "mlevolve-effective.yaml",
        "integration": "explicit course execution/metric hooks; no monkey patches",
        "memory_initialized": search.global_memory is not None,
        "coldstart_guidance_found": cfg.coldstart.description != "None model",
    })
    audit.event("mlevolve_initialized", memory_initialized=search.global_memory is not None)
    return search
