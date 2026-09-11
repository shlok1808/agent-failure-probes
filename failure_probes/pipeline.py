import json
import platform
import time
import subprocess
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import numpy as np
from .common import ROOT, read_json, write_json, digest, iter_episodes, append_jsonl
from .environment import TextCraft, upstream_conversation, parse_action, UPSTREAM, COMMIT

try:  # only needed to free CUDA memory between retries
    import torch
except ImportError:  # pragma: no cover
    torch = None

# A broken setup must stop immediately rather than manufacture 1000 error records.
FATAL_ERRORS = (KeyboardInterrupt, SystemExit, MemoryError, ImportError, OSError)


def require_deterministic_hashing():
    # Upstream builds prompts from `random.sample(list(<set of str>))`, so set
    # iteration order leaks into the frozen task text unless hashing is fixed.
    if sys.flags.hash_randomization:
        raise ValueError("Run with PYTHONHASHSEED=0; recipe prompts depend on set iteration order")


def select_task_ids(env, config, rows):
    selection = config["task_selection"]
    if selection["mode"] == "recipe_depth":
        chosen = env.tasks_by_depth(selection["recipe_depths"])
    elif selection["mode"] == "agenteval_prefix":
        chosen = [int(row["item_id"].split("_")[-1]) for row in rows[:config["num_tasks"]]]
    else:
        raise ValueError(f'Unknown task selection mode: {selection["mode"]}')
    # An assertion, never a truncation: silently slicing would reintroduce the
    # ordering dependence this whole change exists to remove.
    if len(chosen) != selection["expected_num_tasks"]:
        raise ValueError(f'Selection produced {len(chosen)} tasks, expected {selection["expected_num_tasks"]}')
    return chosen


def prepare(config_path, run, task_ids=None, dry_run=False):
    from huggingface_hub import hf_hub_download
    run = Path(run)
    if not dry_run and (run / "manifest.json").exists():
        raise ValueError("Run already exists; use collect to resume, or a new output directory")
    config = read_json(config_path)
    if config["stage"] not in ("debug", "stage2"):
        raise ValueError(f'Unknown stage: {config["stage"]}')
    require_deterministic_hashing()
    if config["stage"] == "stage2" and Path(run).name.startswith("debug"):
        raise ValueError("Debug data must not become confirmatory data; use a non-debug run directory")
    sha = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if sha != COMMIT or subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip():
        raise ValueError("Upstream checkout differs from pinned, unmodified commit")
    env = TextCraft()
    # AgentEval publishes indices only, never goals, so it is provenance and a
    # coverage cross-check - not the selection mechanism.
    file = hf_hub_download("AgentGym/AgentEval", "textcraft_test.json", repo_type="dataset", revision=config["dataset_revision"])
    rows = read_json(file)
    agenteval_indices = [int(row["item_id"].split("_")[-1]) for row in rows]
    if task_ids is None:
        task_ids = select_task_ids(env, config, rows)
    if len(task_ids) != config["num_tasks"] or len(set(task_ids)) != len(task_ids):
        raise ValueError("Expected the configured number of distinct task indices")
    fingerprint = env.fingerprint()
    tasks = [env.freeze_task(i, config["environment_seed"]) for i in task_ids]
    if len({t["goal"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate goals: task indices must not wrap around the catalog")
    manifest = {"schema_version": 2, "config": config, "upstream_commit": sha,
                "recipe_order": fingerprint,
                "agenteval": {"revision": config["dataset_revision"], "n_rows": len(rows),
                              "indices_covered_by_selection": len(set(agenteval_indices) & set(task_ids)),
                              "note": "AgentEval publishes indices only; the goal an index resolves to is "
                                      "tree-order dependent, so this is index coverage, not goal identity."},
                "conversation_start": upstream_conversation(), "tasks": tasks,
                "python": sys.version, "platform": platform.platform(),
                "packages": {p: version(p) for p in ["torch", "transformers", "numpy", "scikit-learn", "gymnasium"]}}
    manifest["manifest_hash"] = digest(manifest)
    if dry_run:
        # Writes nothing: this is the cross-platform determinism check.
        print(json.dumps({"recipe_order": fingerprint, "manifest_hash": manifest["manifest_hash"],
                          "selection_digest": digest([[t["data_idx"], t["goal"], t["task_hash"]] for t in tasks]),
                          "n_tasks": len(tasks),
                          "tasks": [{"data_idx": t["data_idx"], "goal": t["goal"],
                                     "recipe_depth": t["recipe_depth"], "task_hash": t["task_hash"]} for t in tasks]},
                         indent=2))
        return manifest
    write_json(run / "manifest.json", manifest)
    print(f"Prepared {len(tasks)} frozen tasks. Debug data must not become confirmatory data.", flush=True)
    return manifest


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
    require_deterministic_hashing()
    env = TextCraft()
    # Check the environment before the model loads: a mismatch then costs
    # seconds instead of GPU hours.
    if env.fingerprint() != manifest["recipe_order"]:
        raise ValueError("Crafting tree differs from the tree used at prepare time")
    actor = Actor(config, device)
    runtime = {"device": actor.device, "dtype": str(actor.dtype),
                                    "model_revision": actor.model.config._commit_hash,
                                    "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                                    "code_hashes": {str(p.relative_to(ROOT)): digest(p.read_text()) for p in sorted((ROOT / "failure_probes").glob("*.py"))}}
    if (run / "runtime.json").exists():
        prior = read_json(run / "runtime.json")
        for field in ["device", "dtype", "model_revision", "code_hashes"]:
            if prior[field] != runtime[field]:
                raise ValueError(f"Resume changes {field}; use a fresh run directory")
    else:
        write_json(run / "runtime.json", runtime)
    total = len(manifest["tasks"]) * config["attempts_per_task"]
    done = 0
    started = time.time()
    for task in manifest["tasks"]:
        for attempt in range(config["attempts_per_task"]):
            done += 1
            episode_id = f'{task["task_id"]}_attempt_{attempt:02d}'
            output = run / "episodes" / f"{episode_id}.json"
            if output.exists():
                existing = read_json(output)
                if existing["manifest_hash"] != manifest["manifest_hash"]:
                    raise ValueError("Cannot mix different configurations in one run")
                continue
            for retry in range(config["max_episode_retries"] + 1):
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
                        messages += [{"role": "assistant", "content": result["response"]}, {"role": "user", "content": result["observation"]}]
                        if result["done"]:
                            break
                    record["status"] = "complete"
                    record["success"] = bool(record["rounds"][-1]["reward"] == 1)
                    record["termination"] = "success" if record["success"] else "round_limit"
                    record["total_generated_tokens"] = sum(len(r["generated_token_ids"]) for r in record["rounds"])
                    write_json(output, record)
                    rate = (time.time() - started) / done
                    print(f'[{done}/{total}] {episode_id} rounds={len(record["rounds"])} '
                          f'success={record["success"]} eta={(total - done) * rate / 3600:.1f}h', flush=True)
                    append_jsonl(run / "progress.jsonl", {"episode_id": episode_id, "index": done,
                                                          "success": record["success"], "rounds": len(record["rounds"]),
                                                          "seconds": round(time.time() - started, 1)})
                    break
                except FATAL_ERRORS:
                    raise
                except Exception as exc:
                    # Infrastructure faults are not scientific failure labels. Record
                    # every attempt separately, then continue: one transient CUDA
                    # fault must not discard a multi-hour collection. Per-round seeds
                    # are deterministic, so a retry only rescues transient faults -
                    # a deterministic fault recurs and exhausts the budget, by design.
                    record.update({"status": "error", "error": repr(exc), "retry": retry})
                    write_json(run / "errors" / f"{episode_id}.retry{retry}.json", record)
                    print(f'[{done}/{total}] {episode_id} ERROR (retry {retry}): {exc!r}', flush=True)
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()


def extract(run, device=None):
    from .actor import Actor
    run = Path(run)
    manifest = validate_manifest(run)
    actor = Actor(manifest["config"], device)
    runtime = read_json(run / "runtime.json")
    if actor.device != runtime["device"] or str(actor.dtype) != runtime["dtype"]:
        raise ValueError("Replay backend/precision must match collection")
    vectors, normalized, keys, replay_checks = [], [], [], []
    for episode in iter_episodes(run):
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


AUDIT_PURPOSE = {
    "debug": "Debug only; counts and AUC are not scientific evidence",
    "stage2": "Confirmatory run integrity checks; see docs/reproduction.md for scope",
}


def audit(run):
    run = Path(run)
    manifest = validate_manifest(run)
    env = TextCraft()
    if env.fingerprint() != manifest["recipe_order"]:
        raise ValueError("Crafting tree differs from the tree used at prepare time")
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    seen, initial_tokens = set(), {}
    # Single streaming pass: accumulate counters rather than holding every episode.
    n_episodes = n_successes = n_fallbacks = n_token_limit = n_generated = 0
    validity_counter = Counter()
    expected_keys = set()
    for episode in iter_episodes(run):
        n_episodes += 1
        n_successes += bool(episode["success"])
        validity_counter[episode["rounds"][0]["validity"]] += 1
        n_token_limit += sum(r["hit_token_limit"] for r in episode["rounds"])
        n_generated += episode["total_generated_tokens"]
        for r in episode["rounds"][:manifest["config"]["feature_rounds"]]:
            n_fallbacks += r["probe_site"] != "last_action_token"
            expected_keys.add(f'{episode["episode_id"]}:r{r["round"]}')
        if episode["episode_id"] in seen:
            raise ValueError("Duplicate episode")
        seen.add(episode["episode_id"])
        task = tasks[episode["task_id"]]
        assert episode["task_hash"] == task["task_hash"]
        env.reset(task)
        # Digest rather than the raw list: 1000 episodes x ~2000 ints is pure waste.
        first = digest(episode["rounds"][0]["prompt_token_ids"])
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
        assert set(features["keys"].tolist()) == expected_keys
        assert len(features["keys"]) == len(expected_keys)
        assert features["residual"].shape == (len(expected_keys), extracted["hidden_size"])
        assert np.isfinite(features["residual"]).all()
        assert np.isfinite(features["final_normalized"]).all()
        assert max(extracted["repeat_replay_max_abs_error"]) < 1e-5
        feature_check = "passed"
    errors = sorted((run / "errors").glob("*.json")) if (run / "errors").exists() else []
    report = {"purpose": AUDIT_PURPOSE.get(manifest["config"]["stage"], AUDIT_PURPOSE["debug"]),
              "expected_episodes": expected,
              "completed_episodes": n_episodes, "all_episodes_complete": n_episodes == expected,
              "errored_episode_records": [e.name for e in errors],
              "environment_replay": "passed", "identical_initial_prompt_per_task": "passed",
              "recipe_order_verified": "passed",
              "feature_integrity": feature_check,
              "successes": n_successes,
              "first_action_validity": dict(validity_counter),
              "feature_site_fallbacks": n_fallbacks,
              "responses_hitting_token_limit": n_token_limit,
              "generated_tokens": n_generated}
    write_json(run / "audit.json", report)
    print(json.dumps(report, indent=2))
