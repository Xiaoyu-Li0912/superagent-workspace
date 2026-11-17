"""Linux bubblewrap runner: minimal root, no host home/mnt/network or secrets."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class Sandbox:
    def __init__(self, timeout=300, memory_mb=4096, cpus=2):
        if sys.platform != "linux":
            raise RuntimeError("Candidate isolation requires Linux/WSL with bubblewrap")
        self.timeout, self.memory_mb, self.cpus = timeout, memory_mb, cpus
        self.environment = Path(sys.prefix).resolve()
        if self.environment == Path("/usr") or not str(self.environment).startswith("/opt/"):
            raise RuntimeError("Use dedicated /opt virtual environment; never bind a home directory")

    def command(self, code, data, model, output, mode):
        cmd = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--cap-drop", "ALL",
               "--clearenv", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
               "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
               "--ro-bind", str(self.environment), str(self.environment),
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/etc",
               "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
               "--ro-bind", str(code.resolve()), "/solution.py",
               "--ro-bind", str(data.resolve()), "/data",
               "--bind" if mode == "train" else "--ro-bind", str(model.resolve()), "/model",
               "--bind", str(output.resolve()), "/output", "--chdir", "/tmp"]
        for key, value in {"PATH": f"{self.environment}/bin:/usr/bin", "HOME": "/tmp", "LANG": "C.UTF-8",
                           "OPENBLAS_NUM_THREADS": str(self.cpus), "OMP_NUM_THREADS": str(self.cpus),
                           "MKL_NUM_THREADS": str(self.cpus), "PYTHONNOUSERSITE": "1"}.items():
            cmd += ["--setenv", key, value]
        cmd += [str(self.environment / "bin/python"), "-I", "-B", "/solution.py", mode,
                "--data", "/data", "--model", "/model"]
        if mode == "predict":
            cmd += ["--output", "/output/predictions.csv"]
        # Avoid preexec_fn in the parallel scheduler: Python locks inherited by
        # forked children can deadlock before exec. Native tools apply limits.
        allowed = sorted(os.sched_getaffinity(0))[:self.cpus]
        return ["prlimit", f"--as={self.memory_mb * 1024**2}",
                f"--cpu={self.timeout}:{self.timeout + 1}", f"--fsize={512 * 1024**2}",
                "--nofile=256", "--core=0", "--", "taskset", "--cpu-list",
                ",".join(map(str, allowed))] + cmd

    def run(self, code, data, model, output, mode, log):
        model.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        # File redirection plus RLIMIT_FSIZE bounds adversarial output; no unbounded PIPE.
        with log.open("wb") as stream:
            process = subprocess.Popen(self.command(code, data, model, output, mode), stdout=stream,
                                       stderr=subprocess.STDOUT, env={"PATH": "/usr/bin:/bin"},
                                       start_new_session=True)
            try:
                process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise TimeoutError(f"Candidate {mode} exceeded {self.timeout}s")
            finally:
                # Namespace PID 1 exiting kills surviving children as well.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        text = log.read_bytes()[-16000:].decode("utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError(f"Candidate {mode} exited {process.returncode}:\n{text}")
        return time.monotonic() - start, text
