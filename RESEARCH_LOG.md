# Research log

## 2026-09-13 — Stage 2: replication holds, within-task signal does not

**Run:** `runs/stage2-002`. 125 depth-1 TextCraft tasks x 8 attempts = 1000 episodes,
Qwen3-1.7B, A100-SXM4-40GB, 6 sharded workers, 81 minutes, zero errors.
775 successes, 57 mixed-outcome tasks, 0 probe-site fallbacks. Audit re-executed
every logged action against a fresh environment and matched on every field.

### Headline

| Metric | Ruan | Ours |
|---|---|---|
| Pooled cross-fitted AUC (their reported statistic) | 0.81 | **0.791** |
| Within-task pair-weighted AUC (not in the paper) | — | **0.547**, 95% CI [0.46, 0.63] |

**The replication succeeded.** Methodology verified against the paper text directly,
not inferred:

- Fig. 3: "Cross-fitted post-generation probe AUC ... among episodes alive at each
  round." At round 1 all episodes are alive, so the statistic is pooled over episodes.
- Per-Round Failure Scorers: "a logistic regression on standardized features with L2
  regularization and C=1" on "the residual-stream hidden state at the final token of
  the agent's generated action", scored under "task-grouped stratified group k-fold
  cross-fitting".
- Appendix A: "For Qwen3-1.7B, the full-matrix run swept layers {4, 8, …, 28} and
  selected layer 28."

We match on estimator, C, token site, layer, and grouped cross-fitting.

### The finding

The paper reports no within-task statistic. Ours:

| Feature set | Pooled AUC | Within-task AUC |
|---|---|---|
| Hidden state (layer 28) | 0.791 | **0.547** |
| Cheap 5-number surface baseline | 0.698 | **0.615** |
| Hidden + surface | 0.795 | 0.549 |
| Text baseline (TF-IDF + surface) | 0.717 | 0.606 |

Three pieces of evidence that the pooled number is mostly task difficulty:

1. **99.6% of pooled comparisons are cross-task.** 173,748 of 174,375 failure/success
   pairs compare *different* tasks. Only 627 pairs hold the task fixed.
2. **77% of score variance is between tasks, not within.** The probe emits
   approximately one score per task: `flower_pot` ~0.0000, `*_planks` ~0.98.
3. **The surface baseline beats it within-task** (0.615 vs 0.547) while using only
   response length, prompt length and mean action-token logprob — features with no
   access to internal state.

Per-task AUCs are bimodal (12 near 0, 14 near 0.5, 13 near 1.0); fold macro-AUC
swings 0.39–0.64 by held-out split.

**Confident misfires.** `blaze_powder` scores 0.9973 and fails 0/8. `yellow_dye`
scores 0.9130 and fails 1/8. A difficulty detector breaks this way; a trajectory-aware
failure detector should not.

### Illustrative case: `textcraft_15` (brown_dye)

Recipe: `craft 1 brown dye using 1 cocoa beans` — count 1, noun plural.

| Attempts | First action | Rounds | Outcome | Hidden | Surface |
|---|---|---|---|---|---|
| 0–3, 5–7 | `get 1 cocoa beans` | 2 | success | 0.0003 | 0.1146 |
| 4 | `get 1 cocoa **bean**` | 20 | FAIL | **0.0002** | **0.3008** |

One token decides the episode. The hidden probe ranked the failure *below* all seven
successes (task AUC 0.000); the surface baseline caught it. The discriminating
information was in the emitted text, and the layer-28 state missed it.

This is the plural trap: 15 of 125 task recipes name a count-1 plural ingredient
(`1 acacia logs`), where the plural form is a distinct registered item. The model
normalises the grammar, collects the wrong item, and loops to the round cap. These
tasks fail 8/8 and are the bulk of what the probe actually learned to recognise.

### Why within-task matters operationally

Abort simulation on our own scores — threshold the probe, abort above it:

| Threshold | Aborted | Failures caught | Successes killed | Recall kept |
|---|---|---|---|---|
| 0.50 | 169 | 99/225 | 70/775 | 91.0% |
| 0.90 | 119 | 72/225 | 47/775 | 93.9% |
| 0.99 | 46 | 28/225 | 18/775 | 97.7% |

Because the probe scores tasks rather than rollouts, it aborts all 8 attempts of a
task or none. On mixed tasks that is unavoidably lossy: aborting `golden_carrot`
(score 0.9748) catches 2 failures and destroys 6 successes; aborting `blaze_powder`
(0.9973) destroys 8 successes and catches nothing. Within-task discrimination is
precisely the capability that would let a gate keep the good rollouts of a hard task.
Without it, the recall/savings frontier is bounded by task-level granularity.

### Reading

"Internal states predict failure" replicates at the pooled level. But the pooled
statistic conflates *which task this is* with *whether this rollout is going wrong*,
and our decomposition puts most of the weight on the former. Ruan's cascade
application is unaffected — an abort gate does not care why a score is informative —
but the mechanistic claim the framing invites is not supported by our data.

### Deviations from the paper

- 125 tasks x 8 (1000 episodes) vs 100 x 8 (800). Our pool is entirely depth-1, so
  *more* homogeneous in difficulty than theirs likely was — which should make the
  difficulty confound harder to produce, not easier. We still got 0.79.
- Sampling parameters (temp 0.8, top-p 0.95) are ours; the paper does not list them.
- The paper's exact 100-task list is not supplied, and upstream task indices are
  filesystem-load-order dependent (see `AGENTS.md`), so task-level correspondence to
  the paper is not established.

### Open questions

1. Does the within-task collapse hold on WebShop/Qwen3-1.7B (paper: 0.591 at round 1,
   rising to 0.896 by round 3)? Later rounds carry real trajectory evidence, so the
   within-task signal may survive there where it does not at round 1.
2. Does it hold at rounds 2–6? Episodes already store every round; only `extract`
   needs re-running with `feature_rounds > 1`. This is the cheapest next experiment.
3. Would a stronger model (Qwen-2.5-7B, paper 0.86) show genuine within-task signal,
   or a larger version of the same confound?
4. Excluding the 15 plural-trap tasks changes the difficulty mix — worth reporting as
   a sensitivity check, but not as the primary estimate (it drops the tasks the model
   is worst at, which is selection on the outcome).
