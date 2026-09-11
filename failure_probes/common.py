import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def episode_paths(run):
    return sorted((Path(run) / "episodes").glob("*.json"))


def iter_episodes(run):
    """Stream episodes one at a time.

    A 20-round episode is ~0.5 MB of JSON; at 1000 episodes, holding them all as
    Python objects costs several GB. Callers that need a single pass should use
    this rather than episodes().
    """
    for path in episode_paths(run):
        yield read_json(path)


def episodes(run):
    return list(iter_episodes(run))


def append_jsonl(path, value):
    """Append one record. write_json's atomic replace cannot append."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value) + "\n")
