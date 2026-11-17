import json
import shutil
from pathlib import Path

from .audit import file_manifest, write_json


def freeze(candidate: Path, run: Path, result):
    target = run / "best"
    if target.exists():
        raise ValueError("Already frozen")
    target.mkdir()
    shutil.copyfile(candidate / "solution.py", target / "solution.py")
    # Reject symlinks before copying any generated artifact into the trusted package.
    file_manifest(candidate / "model")
    shutil.copytree(candidate / "model", target / "model")
    manifest = {"candidate": result, "files": file_manifest(target), "locked": True}
    write_json(run / "freeze.json", manifest)
    return target


def verify_frozen(run: Path):
    manifest = json.loads((run / "freeze.json").read_text(encoding="utf-8"))
    if manifest["files"] != file_manifest(run / "best"):
        raise ValueError("Frozen code or model has changed")
    return manifest
