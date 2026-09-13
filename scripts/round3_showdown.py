"""Does the hidden state beat what an outside observer can already see?

At round r an observer has: every prompt, every response, every error message so
far. That is the honest competitor to activation probing. If text matches the
hidden state, instrumenting activations buys nothing.

Also breaks the probe's scores down by failure mode, to test whether a
better-placed probe (layer 12) detects the stuck-in-a-loop failures that the
layer-28 round-1 probe missed, or still mostly the lexical plural traps.

    python scripts/round3_showdown.py --run runs/stage2-002 --round 3 --layer 12
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from failure_probes.analysis import grouped_scores, within_task, bootstrap_within
from failure_probes.common import read_json, write_json, iter_episodes
from scripts.failure_modes import classify


def observable_text(episode, index):
    """Everything a watcher outside the model could read by this round."""
    parts = []
    for record in episode["rounds"][:index + 1]:
        parts.append(record["response"])
        parts.append(record["observation"])
    return "\n".join(parts)


def surface(episode, index):
    record = episode["rounds"][index]
    prior = episode["rounds"][:index]
    return [len(record["generated_token_ids"]), len(record["prompt_token_ids"]),
            sum(p["validity"] != "executable" for p in prior),
            sum(p["validity"] == "impossible" for p in prior),
            len({p["action"] for p in prior}) if prior else 0.0]


def replay_layer(actor, record, layer_number):
    ids = record["prompt_token_ids"] + record["generated_token_ids"]
    target = len(record["prompt_token_ids"]) + record["probe_generated_token_index"]
    captured = {}

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["v"] = hidden[0, target].detach().float().cpu().numpy().copy()

    handle = actor.model.model.layers[layer_number - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            actor.model.model(input_ids=torch.tensor([ids], device=actor.device), use_cache=False)
    finally:
        handle.remove()
    return captured["v"]


def report_model(name, y, groups, scores, config):
    mask = np.isfinite(scores)
    stats = within_task(y[mask], scores[mask], groups[mask])
    ci = bootstrap_within(stats["per_task"], config["bootstrap_samples"], config["analysis_seed"])
    pooled = float(roc_auc_score(y[mask], scores[mask]))
    ci_text = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "n/a"
    print(f'{name:34s} within={stats["pair_weighted_auc"]:.3f} pooled={pooled:.3f} ci={ci_text}')
    return {"model": name, "within_task_auc": stats["pair_weighted_auc"], "pooled_auc": pooled,
            "mixed_tasks": stats["mixed_tasks"], "conditional_bootstrap_95": ci}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    from failure_probes.actor import Actor
    run = Path(args.run)
    manifest = read_json(run / "manifest.json")
    config = manifest["config"]
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    index = args.round - 1
    actor = Actor(config, args.device)

    x28, x_layer, texts, surf, y, groups, episodes_kept = [], [], [], [], [], [], []
    for episode in iter_episodes(run):
        if len(episode["rounds"]) <= index:
            continue
        record = episode["rounds"][index]
        x_layer.append(replay_layer(actor, record, args.layer))
        x28.append(replay_layer(actor, record, config["layer_number"]))
        texts.append(observable_text(episode, index))
        surf.append(surface(episode, index))
        y.append(int(not episode["success"]))
        groups.append(episode["task_id"])
        episodes_kept.append(episode)
    y, groups = np.array(y), np.array(groups)
    surf = np.array(surf, dtype=float)
    print(f'round {args.round}: {len(y)} alive, {int(y.sum())} eventual failures\n', flush=True)

    results = []
    hid_scores, splits = grouped_scores(np.stack(x_layer), y, groups,
                                        config["cv_folds"], config["analysis_seed"])
    results.append(report_model(f"hidden layer {args.layer}", y, groups, hid_scores, config))
    s28, _ = grouped_scores(np.stack(x28), y, groups, config["cv_folds"], config["analysis_seed"])
    results.append(report_model(f'hidden layer {config["layer_number"]} (paper)', y, groups, s28, config))
    ssurf, _ = grouped_scores(surf, y, groups, config["cv_folds"], config["analysis_seed"])
    results.append(report_model("surface (observable counts)", y, groups, ssurf, config))

    # Text: everything an outside observer has read by this round, fit per fold.
    text_scores = np.full(len(y), np.nan)
    for split in splits:
        if split["status"] != "ok":
            continue
        train, test = split["train"], split["test"]
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=5000)
        a = vec.fit_transform([texts[i] for i in train])
        b = vec.transform([texts[i] for i in test])
        scaler = StandardScaler().fit(surf[train])
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(hstack([a, csr_matrix(scaler.transform(surf[train]))]), y[train])
        text_scores[test] = clf.predict_proba(
            hstack([b, csr_matrix(scaler.transform(surf[test]))]))[:, 1]
    results.append(report_model("TEXT observable + surface", y, groups, text_scores, config))

    # Failure-mode breakdown for the better-placed probe.
    print(f'\nWhat does layer {args.layer} at round {args.round} actually detect?')
    succ = float(np.mean(hid_scores[(y == 0) & np.isfinite(hid_scores)]))
    print(f'  mean score on eventual successes: {succ:.3f}')
    modes = {}
    for episode, score, label in zip(episodes_kept, hid_scores, y):
        if not label or not np.isfinite(score):
            continue
        for mode in classify(episode, tasks[episode["task_id"]]):
            modes.setdefault(mode, []).append(float(score))
    print(f'  {"failure mode":32s} {"n":>4s} {"mean score":>11s} {">0.5":>6s}')
    for mode, scores in sorted(modes.items(), key=lambda kv: -len(kv[1])):
        share = 100 * sum(s > 0.5 for s in scores) / len(scores)
        print(f'  {mode:32s} {len(scores):4d} {np.mean(scores):11.3f} {share:5.0f}%')

    write_json(run / f"showdown_r{args.round}_l{args.layer}.json",
               {"round": args.round, "layer": args.layer, "n_alive": int(len(y)),
                "device": actor.device, "models": results,
                "mean_score_successes": succ,
                "failure_modes": {k: {"n": len(v), "mean_score": float(np.mean(v)),
                                      "share_above_0.5": float(np.mean([s > 0.5 for s in v]))}
                                  for k, v in modes.items()}})
    print(f'\nwrote {run}/showdown_r{args.round}_l{args.layer}.json')


if __name__ == "__main__":
    main()
