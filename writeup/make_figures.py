"""Figures for the write-up."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path("runs/stage2-002")
OUT = Path("writeup")
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})
FAIL, SUCC = "#c0392b", "#2c7fb8"

# ---------------------------------------------------------------- figure 1
# Per task, mean score on its failures vs on its successes. A probe carrying
# within-task information puts points above the diagonal; a probe that emits one
# score per task puts them on it.
preds = {p["episode_id"]: p for p in json.loads((RUN / "predictions.json").read_text())}
show = json.loads((RUN / "showdown_r3_l12.json").read_text())

by_task = defaultdict(list)
for p in preds.values():
    by_task[p["task_id"]].append(p)
mixed = {k: v for k, v in by_task.items() if 0 < sum(x["failure"] for x in v) < len(v)}

xs = [np.mean([e["hidden"] for e in v if not e["failure"]]) for v in mixed.values()]
ys = [np.mean([e["hidden"] for e in v if e["failure"]]) for v in mixed.values()]

fig, ax = plt.subplots(figsize=(3.2, 2.05))
ax.fill_between([0, 1], [0, 1], [1, 1], color="#1a5276", alpha=0.055, lw=0, zorder=0)
ax.text(0.30, 0.90, "probe ranks failures higher", fontsize=6.2, color="#1a5276", ha="center")
ax.plot([0, 1], [0, 1], color="0.5", lw=0.8, ls="--", zorder=1)
ax.scatter(xs, ys, s=13, c="#1a5276", alpha=0.85, linewidths=0, zorder=3)
above = sum(y > x for x, y in zip(xs, ys))
ax.set_xlabel("mean score on that task's successes")
ax.set_ylabel("mean score on its failures")
ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.set_title(f"Reported config: only {above} of {len(xs)} tasks\nabove the line", fontsize=8, pad=4)
fig.tight_layout()
fig.savefig(OUT / "fig1_confound.pdf", bbox_inches="tight")
print(f"fig1 ok ({above}/{len(xs)} above diagonal)")

# ---------------------------------------------------------------- figure 2
# Layer sweep: the reported layer sits on the downslope.
r1 = json.loads((RUN / "layer_sweep_r1.json").read_text())
r3 = json.loads((RUN / "layer_sweep_r3.json").read_text())
layers = [l["layer"] for l in r1["layers"]]

fig, ax = plt.subplots(figsize=(3.2, 2.05))
ax.axhline(0.5, color="0.7", lw=0.6, ls=":")
ax.text(26.4, 0.508, "chance", fontsize=6.3, color="0.45")
for data, style, label in [(r3, dict(marker="o", color="#1a5276"), "round 3"),
                           (r1, dict(marker="s", color="#7fb3d5"), "round 1")]:
    ax.plot(layers, [l["within_task_auc"] for l in data["layers"]],
            lw=1.3, ms=3.5, label=label, **style)
ax.axvline(28, color=FAIL, lw=0.8, ls="--")
ax.annotate("reported\nlayer", xy=(28, 0.70), xytext=(24.2, 0.60),
            fontsize=6.3, color=FAIL, ha="center",
            arrowprops=dict(arrowstyle="->", color=FAIL, lw=0.55))
ax.set_xlabel("decoder layer")
ax.set_ylabel("within-task AUC")
ax.set_xticks(layers)
ax.set_ylim(0.48, 0.93)
ax.legend(loc="upper left", fontsize=7, ncol=2, columnspacing=1.0)
ax.set_title("Per-rollout signal peaks mid-network", fontsize=8, pad=4)
fig.tight_layout()
fig.savefig(OUT / "fig2_layers.pdf", bbox_inches="tight")
print("fig2 ok")

# ---------------------------------------------------------------- figure 3
# Lexical trap vs structural failure modes.
modes = show["failure_modes"]
rows = [("plural trap (lexical)", "plural_trap_task", 0.77),
        ("repeats one action", "repeated_same_action_5x", 0.29),
        ("mostly invalid actions", "mostly_impossible_actions", 0.29),
        ("format errors", "repeated_format_errors", 0.29),
        ("emits 'abort'", "explicit_abort", 0.29)]
labels = [r[0] for r in rows]
new = [modes[r[1]]["share_above_0.5"] for r in rows]
old = [r[2] for r in rows]

fig, ax = plt.subplots(figsize=(3.6, 1.75))
y = np.arange(len(rows))
ax.barh(y + 0.19, old, height=0.36, color="#bdc3c7", label="round 1, layer 28")
ax.barh(y - 0.19, new, height=0.36, color="#1a5276", label="round 3, layer 12")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("share of failures detected (score > 0.5)")
ax.set_xlim(0, 1.0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=7, handlelength=1.2)

fig.tight_layout()
fig.savefig(OUT / "fig3_modes.pdf", bbox_inches="tight")
print("fig3 ok")
