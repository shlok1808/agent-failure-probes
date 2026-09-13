# Stage 2 result — TextCraft / Qwen3-1.7B, 2026-09-13

1000 episodes (125 depth-1 tasks x 8 attempts), A100, 81 minutes, zero errors.
775 successes, 57 mixed-outcome tasks, 0 probe-site fallbacks.
Audit passed: every logged action re-executed against a fresh environment and matched.

## Headline

| Metric | Ruan | Ours |
|---|---|---|
| Pooled cross-fitted AUC (the paper's reported statistic) | 0.81 | **0.791** |
| Within-task pair-weighted AUC (not reported by the paper) | -- | **0.547** |

**The replication succeeded.** On the statistic Ruan actually reports, we land at
0.791 against their 0.81, with matching methodology (verified against the paper
text, not inferred):

- Figure 3 caption: "Cross-fitted post-generation probe AUC ... among episodes
  alive at each round" -- at round 1 all episodes are alive, so it is pooled
  across episodes, not within-task.
- Per-Round Failure Scorers: "a logistic regression on standardized features with
  L2 regularization and C=1", on "the residual-stream hidden state at the final
  token of the agent's generated action", scored by "task-grouped stratified
  group k-fold cross-fitting".
- Appendix A: "For Qwen3-1.7B, the full-matrix run swept layers {4, 8, ..., 28}
  and selected layer 28."

Our pipeline matches on estimator, C, token site, layer, and grouped cross-fitting.

## The new finding

The paper never computes a within-task statistic. We do, and it changes the
interpretation.

| Feature set | Pooled AUC | Within-task AUC |
|---|---|---|
| Hidden state (layer 28) | 0.791 | **0.547** |
| Cheap 5-number surface baseline | 0.698 | **0.615** |
| Hidden + surface | 0.795 | 0.549 |
| Text baseline (TF-IDF + surface) | 0.717 | 0.606 |

Holding task identity fixed -- comparing failures against successes *on the same
task* -- the hidden-state signal falls to near chance (95% CI [0.46, 0.63],
straddles 0.5), and the surface baseline beats it.

The corroborating detail: the surface baseline reaches pooled AUC 0.698 using only
response length, prompt length and mean action-token logprob. Those have no
privileged access to anything internal; they are correlates of task difficulty.
A large share of the pooled signal arrives with no hidden state involved at all.

Per-task AUCs are bimodal (12 tasks near 0, 14 near 0.5, 13 near 1.0) and fold
macro-AUC swings 0.39-0.64 depending on which tasks are held out -- consistent
with instability rather than a stable moderate effect.

## Reading

"Internal states predict failure" holds at the pooled level and replicates. But
much of that appears to be the probe reading *which task this is* rather than
*whether this particular rollout is going wrong*. The cascade application is
unaffected -- aborting doomed episodes does not care why the score is informative
-- but the mechanistic claim is weaker than the headline implies.

## Deviations from the paper

- 125 tasks x 8 (1000 episodes) vs their 100 x 8 (800). Our pool is all depth-1,
  so more homogeneous in difficulty than theirs likely was, which if anything makes
  the task-difficulty confound harder to produce. We still got 0.79.
- Sampling parameters (temp 0.8, top-p 0.95) are ours; the paper does not list them.
- Task identity: the paper's exact 100-task list is not supplied, and upstream task
  indices are load-order dependent (see AGENTS.md), so task-level correspondence is
  not established.

Raw episodes and features.npz are gitignored; they live in runs/stage2-002/ locally.
