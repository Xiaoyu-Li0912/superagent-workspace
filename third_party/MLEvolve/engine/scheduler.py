# Locally extracted and modified for AIAA3102; see MLEvolve/LOCAL_CHANGES.md.
"""Two-phase scheduling from run.py, reusable with an external execution callback.

Initial drafts are generated sequentially, then deferred execution and search
continuations share a thread pool. Selection/actions remain in AgentSearch.
"""
import time
import logging
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

logger = logging.getLogger("MLEvolve")


def run_search(agent, execute, checkpoint, should_stop=lambda: False, stagger_seconds=10,
               max_workers=None):
    total = agent.acfg.steps
    workers = max_workers or agent.scfg.parallel_search_num
    pending = []
    for _ in range(min(agent.acfg.initial_drafts, total)):
        if should_stop():
            break
        try:
            node = agent.step(None, execute, execute_immediately=False)
            if node is not None:
                pending.append(node)
        except Exception:
            logger.exception("Initial draft generation failed; continuing as in run.py")

    futures = set()
    completed = len(agent.journal) - 1
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        for node in pending:
            if should_stop():
                break
            futures.add(executor.submit(agent.execute_deferred_node, node, execute))
            if stagger_seconds and node is not pending[-1]:
                time.sleep(stagger_seconds)

        def submit(parent=None):
            futures.add(executor.submit(agent.step, parent, execute))

        while len(futures) < min(workers, total - completed) and not should_stop():
            submit()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1)
            for future in done:
                futures.remove(future)
                try:
                    parent = future.result()
                except Exception:
                    logger.exception("Search task failed; continuing from virtual root as in run.py")
                    parent = None
                checkpoint(agent)
                completed = len(agent.journal) - 1
                if completed + len(futures) < total and not should_stop():
                    submit(parent)
        return completed
    except BaseException:
        # Running calls have already reserved budget. Drain them and persist
        # their usage; never freeze while workers can still mutate candidates.
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        checkpoint(agent)
