import hashlib
import json
import os
import time
import threading
from pathlib import Path


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_manifest(root: Path):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink not permitted in artifact: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = sha256(path)
    return result


class Audit:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)

    def event(self, kind, **details):
        record = {"time_unix": time.time(), "event": kind, **details}
        with self.lock, (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
