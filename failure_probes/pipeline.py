import json
import platform
import subprocess
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import numpy as np
from .common import ROOT, read_json, write_json, digest, episodes
from .environment import TextCraft, upstream_conversation, parse_action, UPSTREAM, COMMIT


def prepare(config_path, run, task_ids=None):
    from huggingface_hub import hf_hub_download
    run = Path(run)
    if (run / "manifest.json").exists():
        raise ValueError("Run already exists; use collect to resume, or a new output directory")
    config = read_json(config_path)
    if config["stage"] != "debug":
        raise ValueError("Only the debug stage is enabled. Review results before step 2.")
    sha = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if sha != COMMIT or subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip():
        raise ValueError("Upstream checkout differs from pinned, unmodified commit")
    if task_ids is None:
        file = hf_hub_download("AgentGym/AgentEval", "textcraft_test.json", repo_type="dataset", revision=config["dataset_revision"])
        rows = read_json(file)
        task_ids = [int(row["item_id"].split("_")[-1]) for row in rows[:config["num_tasks"]]]
    if len(task_ids) != config["num_tasks"] or len(set(task_ids)) != len(task_ids):
        raise ValueError("Expected the configured number of distinct task indices")
    env = TextCraft()
    tasks = [env.freeze_task(i, config["environment_seed"]) for i in task_ids]
    if len({t["goal"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate goals: task indices must not wrap around the catalog")
    manifest = {"schema_version": 1, "config": config, "upstream_commit": sha,
                "conversation_start": upstream_conversation(), "tasks": tasks,
                "python": sys.version, "platform": platform.platform(),
                "packages": {p: version(p) for p in ["torch", "transformers", "numpy", "scikit-learn", "gymnasium"]}}
    manifest["manifest_hash"] = digest(manifest)
    write_json(run / "manifest.json", manifest)
    print(f"Prepared {len(tasks)} frozen tasks. Debug data must not become confirmatory data.", flush=True)


def validate_manifest(run):
    manifest = read_json(Path(run) / "manifest.json")
    payload = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if digest(payload) != manifest["manifest_hash"]:
        raise ValueError("Manifest was edited; create a new run")
    return manifest


def collect(run, device=None):
    from .actor import Actor, locate_action_tokens
    run = Path(run)
    manifest = validate_manifest(run)
    config = manifest["config"]
    actor, env = Actor(config, device), TextCraft()
    write_json(run / "runtime.json", {"device": actor.device, "dtype": str(actor.dtype),
                                    "model_revision": actor.model.config._commit_hash,
                                    "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                                    "code_hashes": {str(p.relative_to(ROOT)): digest(p.read_text()) for p in sorted((ROOT / "failure_probes").glob("*.py"))}})
    for task in manifest["tasks"]:
        for attempt in range(config["attempts_per_task"]):
            episode_id = f'{task["task_id"]}_attempt_{attempt:02d}'
            output = run / "episodes" / f"{episode_id}.json"
            if output.exists():
                existing = read_json(output)
                if existing["manifest_hash"] != manifest["manifest_hash"]:
                    raise ValueError("Cannot mix different configurations in one run")
                continue
            observation = env.reset(task)
            messages = list(manifest["conversation_start"]) + [{"role": "user", "content": observation}]
            record = {"episode_id": episode_id, "task_id": task["task_id"], "task_hash": task["task_hash"],
                      "manifest_hash": manifest["manifest_hash"], "attempt": attempt, "rounds": []}
            try:
                for round_index in range(config["max_rounds"]):
                    seed = config["generation_seed"] + task["data_idx"] * 100000 + attempt * 100 + round_index
                    result = actor.generate(messages, seed, keep_logits=round_index < config["feature_rounds"])
                    # Span must refer to the same raw text used for token alignment.
                    _, span, _ = parse_action(result["raw_response"])
                    positions = locate_action_tokens(actor.tokenizer, result["generated_token_ids"], span)
                    result["action_token_indices"] = positions
                    result["probe_generated_token_index"] = positions[-1] if positions else result["last_content_token"]
                    result["probe_site"] = "last_action_token" if positions else "last_response_token_fallback"
                    result.update(env.step(result["response"]))
                    result.update({"round": round_index + 1, "seed": seed})
                    record["rounds"].append(result)
                    print(f'{episode_id} round={round_index+1} action={result["action"]!r} reward={result["reward"]}', flush=True)
                    messages += [{"role": "assistant", "content": result["response"]}, {"role": "user", "content": result["observation"]}]
                    if result["done"]:
                        break
                record["status"] = "complete"
                record["success"] = bool(record["rounds"][-1]["reward"] == 1)
                record["termination"] = "success" if record["success"] else "round_limit"
                record["total_generated_tokens"] = sum(len(r["generated_token_ids"]) for r in record["rounds"])
                write_json(output, record)
            except Exception as exc:
                # Infrastructure faults are not scientific failure labels.
                record.update({"status": "error", "error": repr(exc)})
                write_json(run / "errors" / f"{episode_id}.json", record)
                raise


def extract(run, device=None):
    from .actor import Actor
    run = Path(run)
    manifest = validate_manifest(run)
    actor = Actor(manifest["config"], device)
    vectors, normalized, keys, replay_checks = [], [], [], []
    for episode in episodes(run):
        if episode["manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError("Episode manifest mismatch")
        for record in episode["rounds"][:manifest["config"]["feature_rounds"]]:
            vector, norm = actor.replay(record)
            if not keys:
                again, _ = actor.replay(record)
                replay_checks.append(float(np.max(np.abs(vector - again))))
            vectors.append(vector)
            normalized.append(norm)
            keys.append(f'{episode["episode_id"]}:r{record["round"]}')
    if not keys:
        raise ValueError("No completed episodes to replay")
    np.savez_compressed(run / "features.npz", residual=np.stack(vectors), final_normalized=np.stack(normalized), keys=np.array(keys))
    write_json(run / "extraction.json", {"manifest_hash": manifest["manifest_hash"], "rows": len(keys),
        "hidden_size": len(vectors[0]), "layer_number": actor.layer_number,
        "primary_site": "decoder block output, pre final RMSNorm, one-based layer numbering",
        "device": actor.device, "dtype": str(actor.dtype), "repeat_replay_max_abs_error": replay_checks})
    print(f"Extracted {len(keys)} vectors of dimension {len(vectors[0])}.", flush=True)


def audit(run):
    run = Path(run)
    manifest = validate_manifest(run)
    env = TextCraft()
    data = episodes(run)
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    seen, initial_tokens = set(), {}
    for episode in data:
        if episode["episode_id"] in seen:
            raise ValueError("Duplicate episode")
        seen.add(episode["episode_id"])
        task = tasks[episode["task_id"]]
        assert episode["task_hash"] == task["task_hash"]
        env.reset(task)
        first = episode["rounds"][0]["prompt_token_ids"]
        assert initial_tokens.setdefault(task["task_id"], first) == first, "Same task has different initial tokens"
        for i, record in enumerate(episode["rounds"]):
            actual = env.step(record["response"])
            for field in ["action", "observation", "reward", "done", "inventory_before", "inventory_after", "validity"]:
                assert actual[field] == record[field], f"Environment replay mismatch: {field}"
            assert not record["done"] or i == len(episode["rounds"]) - 1
            assert 0 <= record["probe_generated_token_index"] < len(record["generated_token_ids"])
        assert episode["success"] == bool(episode["rounds"][-1]["reward"] == 1)
    expected = manifest["config"]["num_tasks"] * manifest["config"]["attempts_per_task"]
    feature_check = "not_extracted_yet"
    if (run / "features.npz").exists():
        extracted = read_json(run / "extraction.json")
        assert extracted["manifest_hash"] == manifest["manifest_hash"]
        features = np.load(run / "features.npz", allow_pickle=False)
        expected_keys = {f'{e["episode_id"]}:r{r["round"]}' for e in data for r in e["rounds"][:manifest["config"]["feature_rounds"]]}
        assert set(features["keys"].tolist()) == expected_keys
        assert len(features["keys"]) == len(expected_keys)
        assert features["residual"].shape == (len(expected_keys), extracted["hidden_size"])
        assert np.isfinite(features["residual"]).all()
        assert np.isfinite(features["final_normalized"]).all()
        assert max(extracted["repeat_replay_max_abs_error"]) < 1e-5
        feature_check = "passed"
    report = {"purpose": "Debug only; counts and AUC are not scientific evidence", "expected_episodes": expected,
              "completed_episodes": len(data), "all_episodes_complete": len(data) == expected,
              "environment_replay": "passed", "identical_initial_prompt_per_task": "passed",
              "feature_integrity": feature_check,
              "successes": sum(e["success"] for e in data),
              "first_action_validity": dict(Counter(e["rounds"][0]["validity"] for e in data)),
              "feature_site_fallbacks": sum(r["probe_site"] != "last_action_token" for e in data for r in e["rounds"][:manifest["config"]["feature_rounds"]]),
              "responses_hitting_token_limit": sum(r["hit_token_limit"] for e in data for r in e["rounds"]),
              "generated_tokens": sum(e["total_generated_tokens"] for e in data)}
    write_json(run / "audit.json", report)
    print(json.dumps(report, indent=2))
