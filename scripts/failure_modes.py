"""Categorise what the 225 failed episodes actually did.

"Failure" is one label covering several different behaviours. If the probe only
detects some of them, that matters for interpreting the AUC. Pure text analysis
over logged episodes: no model, no GPU.

    python scripts/failure_modes.py --run runs/stage2-002
"""
import argparse
import re
from collections import Counter, defaultdict

from failure_probes.common import read_json, iter_episodes, write_json


def goal_recipes(task):
    """Recipes in the prompt whose output is the task goal."""
    out = []
    for line in task["commands"].splitlines():
        m = re.match(r"craft (.+?) using (.+)", line)
        if m:
            out.append((line, m.group(1).strip(), m.group(2).strip()))
    return out


def plural_trap(task):
    """Goal recipe names a count-1 plural ingredient (e.g. '1 acacia logs')."""
    for _line, _out, ingredients in goal_recipes(task):
        for part in ingredients.split(","):
            m = re.match(r"(\d+)\s+(.+)", part.strip())
            if m and m.group(1) == "1" and m.group(2).endswith("s"):
                return True
    return False


def classify(episode, task):
    actions = [r["action"] for r in episode["rounds"]]
    validity = [r["validity"] for r in episode["rounds"]]
    labels = []

    if any(a.strip().lower() in {"abort", "quit", "stop", "give up"} for a in actions):
        labels.append("explicit_abort")

    counts = Counter(a for a in actions if a)
    repeated = counts.most_common(1)
    if repeated and repeated[0][1] >= 5:
        labels.append("repeated_same_action_5x")

    if validity.count("executable") == 0:
        labels.append("never_executed_anything")

    if sum(v == "impossible" for v in validity) >= len(validity) * 0.7:
        labels.append("mostly_impossible_actions")

    if sum(v == "incorrectly_formatted" for v in validity) >= 3:
        labels.append("repeated_format_errors")

    if plural_trap(task):
        labels.append("plural_trap_task")

    if not episode["rounds"][-1]["done"] and len(episode["rounds"]) >= 20:
        labels.append("hit_round_cap")

    return labels or ["unclassified"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    manifest = read_json(f"{args.run}/manifest.json")
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    predictions = {p["episode_id"]: p for p in read_json(f"{args.run}/predictions.json")}

    tally = Counter()
    per_label_scores = defaultdict(list)
    rows = []
    n_fail = 0
    for episode in iter_episodes(args.run):
        if episode["success"]:
            continue
        n_fail += 1
        labels = classify(episode, tasks[episode["task_id"]])
        tally.update(labels)
        score = predictions.get(episode["episode_id"], {}).get("hidden")
        for label in labels:
            if score is not None:
                per_label_scores[label].append(score)
        rows.append({"episode_id": episode["episode_id"], "task_id": episode["task_id"],
                     "goal": tasks[episode["task_id"]]["goal"], "labels": labels,
                     "rounds": len(episode["rounds"]), "hidden_score": score})

    print(f"{n_fail} failed episodes\n")
    print(f'{"failure mode":32s} {"count":>6s} {"share":>7s} {"mean probe score":>17s}')
    for label, count in tally.most_common():
        scores = per_label_scores[label]
        mean = sum(scores) / len(scores) if scores else float("nan")
        print(f"{label:32s} {count:6d} {100*count/n_fail:6.1f}% {mean:17.3f}")

    # How well does the probe score each mode? Compare against successes.
    succ = [p["hidden"] for p in predictions.values() if not p["failure"]]
    baseline = sum(succ) / len(succ)
    print(f'\nmean probe score on the 775 SUCCESSES: {baseline:.3f}')
    print("(a mode scoring below this is one the probe ranks as safer than an average success)")

    write_json(f"{args.run}/failure_modes.json",
               {"n_failures": n_fail, "counts": dict(tally),
                "mean_probe_score_by_mode": {k: sum(v)/len(v) for k, v in per_label_scores.items()},
                "mean_probe_score_successes": baseline, "episodes": rows})
    print(f"\nwrote {args.run}/failure_modes.json")


if __name__ == "__main__":
    main()
