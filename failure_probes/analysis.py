"""Held-out-task probes. No cascade or confirmatory testing in debug mode."""
import warnings
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .common import read_json, write_json, episodes


def within_task(labels, scores, tasks):
    labels, scores, tasks = np.asarray(labels), np.asarray(scores), np.asarray(tasks)
    rows = []
    for task in np.unique(tasks):
        ix = tasks == task
        failures, successes = scores[ix & (labels == 1)], scores[ix & (labels == 0)]
        if not len(failures) or not len(successes):
            continue
        comparisons = failures[:, None] - successes[None, :]
        wins = float(np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
        rows.append({"task_id": str(task), "wins": wins, "pairs": comparisons.size, "auc": wins / comparisons.size})
    return {"pair_weighted_auc": sum(r["wins"] for r in rows) / sum(r["pairs"] for r in rows) if rows else None,
            "macro_auc": float(np.mean([r["auc"] for r in rows])) if rows else None,
            "mixed_tasks": len(rows), "per_task": rows}


def bootstrap_within(rows, samples=1000, seed=42):
    if len(rows) < 2:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        selected = [rows[i] for i in rng.integers(len(rows), size=len(rows))]
        values.append(sum(r["wins"] for r in selected) / sum(r["pairs"] for r in selected))
    return np.quantile(values, [0.025, 0.975]).tolist()


def grouped_scores(x, y, groups, folds=5, seed=42):
    scores = np.full(len(y), np.nan)
    splits = []
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        return scores, splits
    cv = StratifiedGroupKFold(n_splits=min(folds, len(np.unique(groups))), shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for train, test in cv.split(x, y, groups):
            assert not set(groups[train]) & set(groups[test])
            split = {"train_tasks": np.unique(groups[train]).tolist(), "test_tasks": np.unique(groups[test]).tolist(),
                     "train": train, "test": test, "status": "ok"}
            if len(np.unique(y[train])) < 2:
                split["status"] = "skipped_one_training_class"
            else:
                probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))
                probe.fit(x[train], y[train])
                scores[test] = probe.predict_proba(x[test])[:, 1]
            splits.append(split)
    return scores, splits


def surface_features(record):
    values = record["token_logprobs"]
    positions = record["action_token_indices"] or [record["last_content_token"]]
    # Round 1 has no previous actions or feedback. Missing history is constant 0.
    return [float(np.mean([values[i] for i in positions])), 0.,
            len(record["generated_token_ids"]), len(record["prompt_token_ids"]), 0.]


def analyze(run):
    from pathlib import Path
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    config = manifest["config"]
    data = episodes(run)
    stored = np.load(run / "features.npz", allow_pickle=False)
    vectors = dict(zip(stored["keys"].tolist(), stored["residual"]))
    x = np.stack([vectors[e["episode_id"] + ":r1"] for e in data])
    y = np.array([int(not e["success"]) for e in data])
    groups = np.array([e["task_id"] for e in data])
    surface = np.array([surface_features(e["rounds"][0]) for e in data])
    hidden_scores, splits = grouped_scores(x, y, groups, config["cv_folds"], config["analysis_seed"])
    score_sets = {"hidden": hidden_scores}
    for name, matrix in [("surface_five", surface), ("hidden_plus_surface", np.column_stack([x, surface]))]:
        score_sets[name], _ = grouped_scores(matrix, y, groups, config["cv_folds"], config["analysis_seed"])
    # Fairer text baseline: task prompt + full response available BEFORE feedback.
    texts = [e["rounds"][0]["prompt"] + e["rounds"][0]["response"] for e in data]
    text_scores = np.full(len(y), np.nan)
    for split in splits:
        if split["status"] != "ok":
            continue
        train, test = split["train"], split["test"]
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=5000)
        a = vectorizer.fit_transform([texts[i] for i in train])
        b = vectorizer.transform([texts[i] for i in test])
        scaler = StandardScaler().fit(surface[train])
        classifier = LogisticRegression(C=1.0, max_iter=2000)
        classifier.fit(hstack([a, csr_matrix(scaler.transform(surface[train]))]), y[train])
        text_scores[test] = classifier.predict_proba(hstack([b, csr_matrix(scaler.transform(surface[test]))]))[:, 1]
    score_sets["prefix_text_plus_surface"] = text_scores
    valid = np.array([e["rounds"][0]["validity"] == "executable" for e in data])
    report = {"stage": config["stage"], "interpretation": "Plumbing checks only; do not interpret debug AUC as evidence",
              "n_episodes": len(y), "n_tasks": len(set(groups)), "successes": int(sum(y == 0)),
              "models": {}, "folds": [{k: v for k, v in s.items() if k not in ("train", "test")} for s in splits],
              "bootstrap_scope": "Conditional on fitted probes and observed mixed-outcome tasks; not retraining uncertainty",
              "pooled_auc_caveat": "OOF probabilities come from different fitted probes. Within-task pairs share a fold.",
              "baseline_caveat": "Beating this text baseline does not prove information is unavailable to all external observers."}
    for name, scores in score_sets.items():
        report["models"][name] = {}
        for subset, mask in [("all", np.ones(len(y), bool)), ("executable_only", valid)]:
            mask &= np.isfinite(scores)
            stats = within_task(y[mask], scores[mask], groups[mask])
            stats.update({"n": int(sum(mask)), "overall_auc": float(roc_auc_score(y[mask], scores[mask])) if len(np.unique(y[mask])) == 2 else None,
                          "conditional_bootstrap_95": bootstrap_within(stats["per_task"], config["bootstrap_samples"], config["analysis_seed"])})
            report["models"][name][subset] = stats
    predictions = [{"episode_id": e["episode_id"], "task_id": e["task_id"], "failure": int(y[i]),
                    **{name: float(s[i]) if np.isfinite(s[i]) else None for name, s in score_sets.items()}} for i, e in enumerate(data)]
    write_json(run / "predictions.json", predictions)
    write_json(run / "analysis.json", report)
    print(f"Saved debug probe diagnostics; {len(y)} episodes, {sum(y == 0)} successes. No confirmatory conclusions.")
