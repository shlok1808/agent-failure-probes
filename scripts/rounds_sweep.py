"""Within-task probe AUC across gate rounds 1..R.

Standalone on purpose. `extract`/`audit` compare hashes of every
failure_probes/*.py against runtime.json, so editing the package would lock us
out of an existing run. This imports the package without modifying it.

Population at round r is "episodes alive at round r" (>= r rounds logged), which
is the paper's convention. Successful episodes terminate early, so later rounds
are progressively smaller and more failure-heavy.

Caveat carried into the output: token_logprobs are stored only for rounds within
the collected `feature_rounds` (1 here), so the round-1 five-feature surface
baseline is not reconstructible at r > 1. We use a logprob-free surface baseline
instead, and label it as such.

    python scripts/rounds_sweep.py --run runs/stage2-002 --rounds 6
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from failure_probes.analysis import grouped_scores, within_task, bootstrap_within
from failure_probes.common import read_json, write_json, iter_episodes


def surface_no_logprob(episode, index):
    """Surface features computable at any round.

    Deliberately excludes mean action-token logprob (unavailable past the
    collected feature_rounds) and adds history counts, which do not exist at
    round 1 but are genuine signal later.
    """
    record = episode["rounds"][index]
    prior = episode["rounds"][:index]
    return [
        len(record["generated_token_ids"]),
        len(record["prompt_token_ids"]),
        sum(p["validity"] != "executable" for p in prior),
        sum(p["validity"] == "impossible" for p in prior),
        float(index),
    ]


def collect_round(run, round_number, actor, vector_kind):
    """Replay one round for every episode alive at it."""
    x, y, groups, surface, ids = [], [], [], [], []
    for episode in iter_episodes(run):
        index = round_number - 1
        if len(episode["rounds"]) <= index:
            continue  # episode already terminated: not alive at this round
        residual, normalized = actor.replay(episode["rounds"][index])
        x.append(normalized if vector_kind == "final_normalized" else residual)
        y.append(int(not episode["success"]))
        groups.append(episode["task_id"])
        surface.append(surface_no_logprob(episode, index))
        ids.append(f'{episode["episode_id"]}:r{round_number}')
    return (np.stack(x), np.array(y), np.array(groups), np.array(surface, dtype=float), ids)


def evaluate(x, y, groups, config, label):
    scores, splits = grouped_scores(x, y, groups, config["cv_folds"], config["analysis_seed"])
    mask = np.isfinite(scores)
    stats = within_task(y[mask], scores[mask], groups[mask])
    from sklearn.metrics import roc_auc_score
    stats["n"] = int(mask.sum())
    stats["pooled_auc"] = (float(roc_auc_score(y[mask], scores[mask]))
                           if len(np.unique(y[mask])) == 2 else None)
    stats["conditional_bootstrap_95"] = bootstrap_within(
        stats["per_task"], config["bootstrap_samples"], config["analysis_seed"])
    stats.pop("per_task")
    stats["model"] = label
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--vector", choices=["residual", "final_normalized"], default="residual")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from failure_probes.actor import Actor
    run = Path(args.run)
    manifest = read_json(run / "manifest.json")
    config = manifest["config"]
    actor = Actor(config, args.device)
    print(f'layer {actor.layer_number}, vector={args.vector}, device={actor.device}', flush=True)

    report = {"run": str(run), "layer_number": actor.layer_number, "vector": args.vector,
              "device": actor.device, "dtype": str(actor.dtype),
              "population": "episodes alive at each round (paper convention)",
              "surface_caveat": "logprob-free surface baseline; token_logprobs exist only for "
                                f'the collected feature_rounds={config["feature_rounds"]}',
              "rounds": []}

    for r in range(1, args.rounds + 1):
        x, y, groups, surface, _ = collect_round(run, r, actor, args.vector)
        if len(np.unique(y)) < 2:
            print(f"round {r}: only one outcome class among survivors; stopping", flush=True)
            break
        entry = {"round": r, "n_alive": int(len(y)), "n_fail": int(y.sum()),
                 "models": [evaluate(x, y, groups, config, "hidden"),
                            evaluate(surface, y, groups, config, "surface_no_logprob")]}
        report["rounds"].append(entry)
        for m in entry["models"]:
            ci = m["conditional_bootstrap_95"]
            ci_text = f'[{ci[0]:.2f}, {ci[1]:.2f}]' if ci else "n/a"
            print(f'round {r:2d} | alive {entry["n_alive"]:4d} | {m["model"]:18s} '
                  f'within={m["pair_weighted_auc"]:.3f} pooled={m["pooled_auc"]:.3f} '
                  f'mixed={m["mixed_tasks"]:2d} ci={ci_text}', flush=True)

    out = Path(args.out) if args.out else run / f"rounds_sweep_{args.vector}.json"
    write_json(out, report)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
