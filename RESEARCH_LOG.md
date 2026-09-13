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

---

## 2026-09-13 (later) — Round and layer sweeps: the earlier conclusion was too strong

**This entry revises the one above.** That entry concluded the within-task signal "does
not hold". That was measured at a single point — round 1, layer 28 — and treated as if it
characterised the model. Sweeping rounds and layers shows the signal is real; the paper's
configuration is simply the weakest corner of the space we searched.

### Round sweep (layer 28, `scripts/rounds_sweep.py`)

Population is episodes alive at each round, the paper's convention.

| Round | Alive | Mixed tasks | Hidden within-task | Surface within-task |
|---|---|---|---|---|
| 1 | 1000 | 57 | 0.585 | 0.607 |
| 2 | 1000 | 57 | 0.650 | 0.691 |
| **3** | 568 | 50 | **0.749** CI [0.68, 0.82] | 0.702 |
| 4 | 343 | 39 | 0.619 | 0.746 |
| 5 | 299 | 33 | 0.534 | 0.486 |
| 6 | 284 | 28 | 0.699 | 0.579 |

Within-task AUC climbs to 0.749 at round 3, with an interval clear of chance. Once the
agent has acted twice and seen the results, the hidden state carries genuine
rollout-specific information. Rounds 4–6 zigzag with intervals spanning 0.3; pairs fall
from 627 at round 1 to 151 at round 6, so nothing should be read into that shape.

### Layer sweep (`scripts/layer_sweep.py`, all layers in one forward pass)

| Layer | R1 within | R1 pooled | R3 within | R3 pooled |
|---|---|---|---|---|
| 4 | 0.620 | 0.722 | 0.718 | 0.697 |
| 8 | 0.640 | 0.802 | 0.721 | 0.790 |
| **12** | **0.651** | **0.830** | **0.801** | 0.855 |
| 16 | 0.625 | 0.807 | 0.795 | **0.867** |
| 20 | 0.587 | 0.780 | 0.766 | 0.854 |
| 24 | 0.563 | 0.758 | 0.755 | 0.841 |
| **28** (paper) | 0.585 | 0.781 | 0.749 | 0.840 |

**Layer 12 beats layer 28 at both rounds on both metrics.** At round 1 it wins even on
the paper's own selection criterion, pooled AUC: 0.830 against 0.781.

### Revised reading

| Configuration | Within-task AUC |
|---|---|
| Round 1, layer 28 — the paper's setup | 0.585, CI [0.51, 0.66] |
| Round 1, layer 12 | 0.651, CI [0.58, 0.72] |
| **Round 3, layer 12** | **0.801, CI [0.74, 0.86]** |

Rollout-specific failure signal exists. At the paper's coordinates it is weak enough to be
mistaken for task difficulty, which is what the first entry concluded. Two rounds later and
sixteen layers earlier it is unambiguous: same task, same recipe, and the probe separates
the attempt that fails from the attempts that succeed.

The finding about pooled AUC still stands — 99.6% of its comparisons are cross-task, and the
probe remains largely a plural-trap detector at round 1 (below). What does not stand is the
inference from that to "the hidden state carries no within-task information".

### Why this is not a multiple-comparisons artifact

Seven layers were searched, so the maximum is biased upward. Three things argue against
that explanation:

1. The layer curve is smooth and unimodal at both rounds, peaking mid-network and declining
   through 28. Independent noise across seven estimates does not organise that way.
2. **Layer 12 wins at round 1 and round 3 independently** — replication within our own data.
3. Within-task and pooled metrics peak at adjacent layers (12 and 16), agreeing on a
   mid-network optimum.

Round 3 / layer 12 remains a selected cell and should be treated as a hypothesis for fresh
data, not a point estimate to quote.

### Failure modes (`scripts/failure_modes.py`)

All 225 failures hit the 20-round cap; none failed fast. 94.7% repeat one action five or
more times; 18 episodes emit a literal `abort`.

| Population | Mean round-1 probe score | Share scoring > 0.5 |
|---|---|---|
| Successes (775) | 0.108 | — |
| Failures on plural-trap tasks (69) | 0.747 | 77% |
| All other failures (156) | 0.310 | 29% |

At round 1, layer 28, the probe is substantially a detector for one lexical, task-level
failure mode. Whether that remains true at round 3 / layer 12 is not yet measured, and is
the obvious next check.

### Caveats

- **Hardware.** Sweeps ran on the Mac (MPS); the canonical `features.npz` was extracted on
  CUDA. Round 1 layer 28 reads 0.547 from CUDA and 0.585 from MPS. Sweeps are internally
  consistent and should be read as relative across rounds and layers; the CUDA numbers stay
  authoritative for the headline. That a vendor swap moves the estimate by 0.04 is itself a
  note about fragility.
- **Surface baseline.** `token_logprobs` are stored only within the collected
  `feature_rounds=1`, so rounds 2+ use a logprob-free variant with history counts. It is not
  the same baseline as the round-1 five-feature version.
- Later rounds are lower-powered and increasingly failure-weighted (79% by round 6).

### Next

1. Failure-mode breakdown at round 3 / layer 12 — does it still track the plural trap, or
   does it detect the stuck-in-a-loop cases the round-1 probe missed?
2. Re-extract the canonical features at layer 12 on CUDA to get an authoritative number.
3. Rounds sweep at layer 12 rather than 28 — the rounds and layers were swept separately,
   so their interaction is only measured at rounds 1 and 3.

---

## 2026-09-13 (later still) — Round 3 / layer 12 showdown

`scripts/round3_showdown.py`, 568 episodes alive at round 3, 225 eventual failures.
Two questions: does the hidden state beat what an outside observer can already read,
and does a better-placed probe detect failure modes the round-1 probe missed.

### Against the observable baseline

| Model | Within-task | Pooled | CI |
|---|---|---|---|
| **Hidden, layer 12** | **0.801** | 0.855 | [0.74, 0.86] |
| Hidden, layer 28 (paper) | 0.749 | 0.840 | [0.68, 0.82] |
| Text observable + surface | 0.725 | 0.825 | [0.64, 0.82] |
| Surface counts only | 0.698 | 0.741 | [0.62, 0.78] |

The text model saw every prompt, response and error message through round 3 — the
honest competitor to activation probing. Layer 12 beats it by 0.076.

**Do not overstate this.** The intervals overlap heavily, and this is evidence of an
increment, not proof of one. A paired test on shared folds, or a replication, is what
would settle it. The text baseline reaching 0.725 on its own is itself the finding that
most of this is readable from the outside.

### What layer 12 detects (mean score on eventual successes: 0.195)

| Failure mode | n | mean score | caught (>0.5) | R1/L28 caught |
|---|---|---|---|---|
| plural_trap_task | 69 | 0.874 | 88% | 77% |
| repeated_same_action_5x | 213 | 0.741 | 73% | ~29%* |
| mostly_impossible_actions | 148 | 0.726 | 71% | ~29%* |
| repeated_format_errors | 56 | 0.798 | 80% | ~29%* |
| explicit_abort | 18 | 0.779 | 83% | ~29%* |
| never_executed_anything | 29 | 0.528 | 48% | ~29%* |

<sub>*the round-1 layer-28 probe caught 29% of non-trap failures as an undifferentiated group.</sub>

**This is the strongest result in the project so far, and it is qualitative rather than
numerical.** The round-1 layer-28 probe was substantially a detector for one lexical,
task-level trap. At round 3 / layer 12 the probe detects every failure mode: looping
jumps from roughly 29% to 73% across 213 episodes, format errors to 80%, explicit
give-ups to 83%. A change in *which categories* are detected is much harder to produce
by chance than a change in a single AUC.

The residual blind spot is `never_executed_anything` at 48% — episodes that never land
a valid action at all. Those may look confused rather than stuck.

### Status of the claim

Defensible now:
- layer 12 > layer 28, replicated at rounds 1 and 3, both metrics, smooth unimodal curve
- the failure-mode generalisation above

Not yet established:
- 0.801 as an effect size. It is a cell selected from a 7-layer x 6-round search and
  should be quoted as an exploratory maximum, not a measurement.
- the margin over the text baseline (overlapping intervals)
- anything beyond TextCraft and Qwen3-1.7B

### Next, in order of value

1. **Pre-registered replication.** Fresh 1000 episodes at a new `generation_seed`, with
   layer 12 / round 3 fixed in advance. Converts a discovered cell into a tested
   prediction. A few GPU-hours.
2. **Re-extract layer 12 / round 3 on CUDA** through the standard `extract`/`analyze`
   path, removing the MPS caveat and producing an authoritative number. ~15 min.
3. Paired fold-level test of hidden vs text, rather than comparing two intervals.
4. Rounds sweep *at layer 12* — rounds and layers were swept separately, so their
   interaction is only observed at rounds 1 and 3.
