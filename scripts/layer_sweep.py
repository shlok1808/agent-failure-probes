"""Within-task probe AUC across layers, at one round.

Ruan chose layer 28 for Qwen3-1.7B by sweeping *pooled* probe AUC. Our result is
that pooled AUC is largely task difficulty, so the layer that maximises it is not
necessarily the layer carrying rollout-specific signal. This sweeps both metrics.

Every layer is captured in ONE forward pass per episode, so N layers cost roughly
the same as one. Standalone for the same reason as rounds_sweep.py: editing
failure_probes/*.py would invalidate the run's recorded code hashes.

    python scripts/layer_sweep.py --run runs/stage2-002 --round 1
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from failure_probes.analysis import grouped_scores, within_task, bootstrap_within
from failure_probes.common import read_json, write_json, iter_episodes


def replay_all_layers(actor, record, layers):
    """One forward pass; capture the probe token at every requested layer."""
    ids = record["prompt_token_ids"] + record["generated_token_ids"]
    target = len(record["prompt_token_ids"]) + record["probe_generated_token_index"]
    captured, handles = {}, []

    def make_hook(layer_number):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_number] = hidden[0, target].detach().float().cpu().numpy().copy()
        return hook

    try:
        for layer_number in layers:
            handles.append(actor.model.model.layers[layer_number - 1]
                           .register_forward_hook(make_hook(layer_number)))
        with torch.inference_mode():
            actor.model.model(input_ids=torch.tensor([ids], device=actor.device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    for layer_number in layers:
        if not np.isfinite(captured[layer_number]).all():
            raise ValueError(f"Non-finite hidden state at layer {layer_number}")
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--layers", default="4,8,12,16,20,24,28",
                        help="comma-separated, one-based (matches the paper's convention)")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    from failure_probes.actor import Actor
    from sklearn.metrics import roc_auc_score

    run = Path(args.run)
    config = read_json(run / "manifest.json")["config"]
    layers = [int(v) for v in args.layers.split(",")]
    actor = Actor(config, args.device)
    n_layers = len(actor.model.model.layers)
    if max(layers) > n_layers:
        raise ValueError(f"model has {n_layers} layers; asked for {max(layers)}")
    print(f"device={actor.device} layers={layers} round={args.round}", flush=True)

    index = args.round - 1
    per_layer, y, groups = {n: [] for n in layers}, [], []
    for episode in iter_episodes(run):
        if len(episode["rounds"]) <= index:
            continue
        captured = replay_all_layers(actor, episode["rounds"][index], layers)
        for layer_number in layers:
            per_layer[layer_number].append(captured[layer_number])
        y.append(int(not episode["success"]))
        groups.append(episode["task_id"])
    y, groups = np.array(y), np.array(groups)
    print(f"replayed {len(y)} episodes alive at round {args.round}", flush=True)

    report = {"run": str(run), "round": args.round, "device": actor.device,
              "n_episodes": int(len(y)), "layers": []}
    for layer_number in layers:
        x = np.stack(per_layer[layer_number])
        scores, _ = grouped_scores(x, y, groups, config["cv_folds"], config["analysis_seed"])
        mask = np.isfinite(scores)
        stats = within_task(y[mask], scores[mask], groups[mask])
        ci = bootstrap_within(stats["per_task"], config["bootstrap_samples"], config["analysis_seed"])
        pooled = float(roc_auc_score(y[mask], scores[mask]))
        report["layers"].append({"layer": layer_number, "within_task_auc": stats["pair_weighted_auc"],
                                 "macro_auc": stats["macro_auc"], "pooled_auc": pooled,
                                 "mixed_tasks": stats["mixed_tasks"], "conditional_bootstrap_95": ci})
        ci_text = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "n/a"
        print(f'layer {layer_number:2d} | within={stats["pair_weighted_auc"]:.3f} '
              f'pooled={pooled:.3f} ci={ci_text}', flush=True)

    best_within = max(report["layers"], key=lambda r: r["within_task_auc"])
    best_pooled = max(report["layers"], key=lambda r: r["pooled_auc"])
    report["best_by_within_task"] = best_within["layer"]
    report["best_by_pooled"] = best_pooled["layer"]
    print(f'\nbest layer by WITHIN-TASK: {best_within["layer"]} ({best_within["within_task_auc"]:.3f})')
    print(f'best layer by POOLED     : {best_pooled["layer"]} ({best_pooled["pooled_auc"]:.3f})'
          '   <- the criterion the paper used')
    write_json(run / f"layer_sweep_r{args.round}.json", report)
    print(f'wrote {run}/layer_sweep_r{args.round}.json')


if __name__ == "__main__":
    main()
