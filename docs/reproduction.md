# Reproduction contract

Sources: [paper PDF](https://arxiv.org/pdf/2607.06503), [v2 methods](https://arxiv.org/html/2607.06503v2), [AgentGym](https://github.com/WooooDyy/AgentGym), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). Checked September 11, 2026.

The immediate target is the first-round TextCraft/Qwen3 probe, not every model/environment cell or the calibrated abort cascade. Ruan reports first-round AUC 0.81 for this cell. Debug AUC on 15 episodes is not a reproduction result.

## Explicitly reported

- TextCraft from AgentGym; Qwen3-1.7B with thinking disabled.
- Main TextCraft cell: 100 tasks, eight rollouts each, 20-round cap.
- Teacher-forced replay of logged trajectories, final generated-action token.
- Selected layer 28; L2 logistic regression, C=1, standardized features.
- Task-grouped stratified cross-fitting; failure is the eventual episode outcome.
- Cheap five-feature surface comparator, plus activation/surface stacking.
- Full cascade gates rounds 1–6; calibration/validation/test proportions 20/20/60 and 20 random-seed repetitions.

## Our documented choices and remaining uncertainty

| Detail | Debug implementation | Status |
|---|---|---|
| AgentGym revision | `3ef9235d23e68e7c2920c5422ad957dc8ced5c6c` | Pinned current source; author's revision unspecified |
| Model revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Pinned public checkpoint; exact author revision unspecified |
| Evaluation task source | AgentEval `1f090d52ef01a83a889da02699e34a500e35a274`, first five TextCraft test indices | Debug choice; author's exact 100-task list not supplied |
| Prompt and parsing | Read standard AgentGym conversation, reproduce its Action extraction/sanitization | Closest public harness, not confirmed author's prompt |
| Task reset | Freeze complete goal/recipe/distractor prompt in manifest | Required for genuinely identical-task comparisons |
| Upstream ordering | Raw source uses sets and filesystem order | Seed alone does not guarantee cross-machine prompts; reuse frozen manifest |
| Model runner | Transformers, sequential, BF16 on MPS/CUDA; CPU float32 | Hardware/backend difference; precision not established as matched |
| Sampling | Temperature .8, top-p .95, top-k 0, repetition penalty 1 | Chosen defaults; paper does not list these |
| Response length | Max 256 new tokens, including visible Thought | Chosen cap; every truncation logged |
| Context | 8192-token safety cap; errors rather than silent truncation | Chosen cap; not a reported paper setting |
| Layer | Block 28 output before final RMSNorm, one-based numbering | Paper ambiguous; normalized output also retained |
| Token site | Last non-whitespace token overlapping Action field | Explicit convention; EOS/trailing text excluded |
| No parseable Action | Last non-special response token fallback, flagged | Required to retain format failures; sensitivity exclusion needed later |
| Raw token likelihood | Unsampled model log probability, not temperature/top-p warped | Paper does not specify raw vs sampled convention |
| Surface count | Full response token count; Action-span mean logprob | Generated-token scope not fully specified in paper |
| Missing history at round 1 | Mean previous logprob and prior errors set to zero | Explicit neutral missing-history convention |
| Cross-validation | 5 stratified group folds, seed 42, no layer/hyperparameter search | Fold count not supplied; debug only one seed |
| Debug scale | 5 tasks × 3 attempts | Deliberate small plumbing test |

## Changes from the earlier discussion

1. Prompt identity must include distractor recipes and their order, not just goal ID.
2. Non-thinking mode still permits the standard harness's visible Thought field.
3. A last generated-action token is not the EOS token. Hooks capture the chosen token explicitly.
4. Upstream action acceptance is permissive. Record actual behavior rather than replacing it with a stricter parser that changes the actor's task.
5. Do not call executable actions productive without an independent, checked productivity definition.
6. AUC near chance does not establish pure difficulty. A stronger probe beating a weak external baseline does not establish information-theoretic necessity of internal access.

## Scope after debug review

For step 2, decide on missing author details and freeze the protocol before collecting fresh 100 × 8 data. Enable round 1 initially, or extend stored features to rounds 1–6 if reproducing the full AUC curve. Multiple seeds should assess split variance. Gate calibration, cascade allocation, and certification are separate work and are not silently claimed by this implementation.

Later hypothesis tests use within-task pairs with half credit for ties, equal-task and pair-weighted averages, validity subset comparisons using the same original probe, and a richer prefix-only baseline. Repeated debug inspection means this dataset cannot provide confirmatory evidence.
