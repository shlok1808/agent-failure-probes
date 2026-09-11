# Debug run completed — September 11, 2026

The collection → replay → probe → within-task comparison pipeline completed. This is a software check, not a reproduction of the paper's result or evidence for H1/H2.

- Setup: pinned AgentGym TextCraft; pinned Qwen3-1.7B, non-thinking; Apple M4 MPS, BF16; five frozen tasks × three attempts; maximum 20 rounds. Exact parameters and source revisions are in `configs/debug.json` and `docs/reproduction.md`.
- Outcomes: **15 completed episodes, 11 successes, four failures**, 7,360 generated tokens. No responses reached the 256-token generation cap.
- First actions: eight executable, six impossible, one incorrectly formatted. All five frozen tasks are independently solvable using their supplied recipes.
- Extracted **15 × 2,048** hidden-state features at the last token of the first action, decoder block 28 before final normalization. No fallback token locations. Repeating the first replay gave maximum absolute difference 0.
- Audit passed: every environment transition and terminal label reproduced, initial prompts matched across attempts at each task, and feature integrity checks passed.
- **13 local tests passed**, covering environment resets, native action parsing, frozen-task solvability, token alignment, replay, grouped splits and within-task metrics.

## What the miniature analysis tells us

Only **one task** had both successes and failures, giving just **two failure–success pairs**. The hidden probe's within-task AUC was 0.25; its pooled held-out-task AUC was approximately 0.17. These are debug diagnostics, not meaningful estimates: five training/test task groups and one mixed task cannot establish reproduction, support H1, or reject it. The executable-only comparison also has only that same mixed task. All five cross-validation folds trained successfully without sharing tasks between training and test.

The small run exposes practical error patterns worth inspecting later, including repeatedly crafting without collecting required ingredients and singular/plural item-name mismatches. They do not establish what the probe reads.

## Bugs caught and fixed

- A generated string could re-tokenize differently from its original generated token IDs. Action-token alignment now uses the original token sequence, with a regression test. The interrupted `debug-001` run is excluded from analysis. Six completed episodes reproduced exactly at the token level on the fresh run after the fix.
- The upstream environment's numeric task indices depend on filesystem ordering. A solvability test passed locally but failed on Linux because it regenerated different tasks. The test now loads the exact frozen debug tasks; the collection pipeline already saves and reuses frozen task definitions.

## Saved state and next boundary

Raw episodes, manifest, extraction checks, vectors, predictions and analysis are saved locally under `runs/debug-002/`; large runtime artifacts and model weights are excluded from Git. The actual five task definitions are tracked in `tests/fixtures/debug_tasks.json`.

**Step 2 has not started.** Before scaling, review the reproduction assumptions, especially the original paper's unspecified prompt/sampling details and task identity mapping. Debug defaults intentionally reject a larger-stage collection. The full routing/abort cascade and causal interventions are outside this initial implementation.

Source: [Ruan et al., Doomed from the Start](https://arxiv.org/abs/2607.06503). See the reproduction notes for the distinction between reported settings and implementation choices.
