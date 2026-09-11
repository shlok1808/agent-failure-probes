# Agent failure probes

Independent, documented reproduction of the **TextCraft / Qwen3-1.7B early failure probe** from [Ruan et al., Doomed from the Start (v2)](https://arxiv.org/html/2607.06503v2).

**Current stage: debug only — 5 tasks × 3 attempts.** The 800-episode reproduction and intervention experiments have not been started. This is not the authors' code or a reproduction of the full abort cascade.

## What this builds

1. Run the real, pinned AgentGym TextCraft mechanics with Qwen3-1.7B.
2. Save complete rollouts, exact model-input tokens, generated tokens, seeds, rewards, and inventory transitions.
3. Replay saved tokens to read the first action's final-token activation.
4. Train a standardized logistic probe with entire tasks held out.
5. Produce debug-only overall and within-task AUC, including executable-action subsets and visible-text baselines.

## Setup

```bash
git clone --recurse-submodules https://github.com/shlok1808/agent-failure-probes.git
cd agent-failure-probes
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Python 3.11+; CUDA, Apple MPS, or CPU. The actor automatically chooses an available accelerator. Model weights download from Hugging Face (several GB). Exact tested package versions are recorded in each run manifest. The local debug environment reuses preinstalled packages through a venv with `--system-site-packages`; a fresh installation does not require this.

## Run the debug experiment

```bash
PYTHONHASHSEED=0 failure-probes prepare --config configs/debug.json --run runs/debug-001
failure-probes collect --run runs/debug-001
failure-probes audit --run runs/debug-001
failure-probes extract --run runs/debug-001
failure-probes analyze --run runs/debug-001
```

- `prepare`: selects tasks by recipe depth (config `task_selection`) under a pinned recipe order, freezes their exact prompts, and records a crafting-tree fingerprint; does not use outcome labels to select tasks. `--dry-run` prints the selection and writes nothing.
- `collect`: resumes completed episodes. A failed interrupted episode restarts from its saved per-step seeds; infrastructure errors are stored separately and are never counted as task failures.
- `audit`: reexecutes every logged command in TextCraft and checks rewards, inventory, and identical initial model inputs for repeated attempts.
- `extract`: stores one block-output vector per episode plus a final-normalized vector for provenance. Reads no future environment feedback into the first-round feature.
- `analyze`: fits standardization, logistic probes, and the richer text baseline on training tasks only. One-class folds return missing scores rather than invented AUC.

To rerun with different settings, create a new run directory. Do not edit an existing manifest. Debug episodes are excluded from later confirmatory data.

## Read the outputs

Inside `runs/debug-001/`:

| File | Meaning |
|---|---|
| `manifest.json` | Frozen configuration, task prompts, revisions, dependencies |
| `runtime.json` | Actual hardware backend, precision, resolved model commit |
| `episodes/*.json` | Full, resumable per-episode logs |
| `errors/*.json` | Infrastructure exceptions, not scientific failures |
| `audit.json` | Environment replay and collection checks |
| `features.npz` | Activation vectors with explicit episode/round keys |
| `extraction.json` | Layer convention and repeated replay check |
| `predictions.json` | Out-of-fold scores and labels |
| `analysis.json` | Debug-only diagnostics, mixed-task counts, fold membership |

Large data and weights are ignored by Git. This repository contains no meeting notes or private correspondence.

## What is and is not matched

Read [docs/reproduction.md](docs/reproduction.md) before interpreting any result. Exact task prompts, rollout sampling parameters, upstream commit, layer indexing/normalization, and fold count are not fully specified by Ruan; our choices are recorded rather than described as exact replication.

The primary probe reads the output of decoder block 28 **before final RMSNorm**, with blocks numbered 1–28. The paper does not specify that convention. Both that vector and final-normalized output are saved. The primary probe always uses the former, not whichever scores better.

The standard AgentGym prompt asks for `Thought:` and `Action:`. Qwen's dedicated thinking mode is disabled, but the visible `Thought:` text remains part of its generated response. The text baseline sees that entire response and the preceding prompt.

## Scientific limits

- Within-task AUC above chance supports rollout-specific prediction; it does not establish a causal failure mechanism.
- A signal drop after filtering does not prove formatting caused the original effect: the sample and difficulty mix change too.
- Beating a finite text classifier does not prove hidden information is fundamentally unavailable to external observers.
- Executable means the environment accepted an action, not that it was useful. No unsupported good/bad-action labels are manufactured.
- First-round extracted features come before first-action feedback. Validity is a **post-hoc evaluation filter**, not an input to the deployed probe or text baseline.
- Bootstrap intervals are conditional on fitted cross-validation probes and observed mixed-outcome tasks; they omit probe-training uncertainty.
- Stage 2 (`configs/stage2.json`, 125 depth-1 tasks x 8 attempts) is now enabled but **not yet collected**. See `AGENTS.md` before running it.
- Recipe load order is pinned to `sorted(os.listdir)`. This is arbitrary but deliberate: it fixes which goal each `data_idx` resolves to, and it changes which items are craftable. See `AGENTS.md`.

## Upstream attribution

TextCraft code and data are loaded from the unchanged [AgentGym](https://github.com/WooooDyy/AgentGym) Git submodule, originally based on [ADaPT](https://github.com/archiki/ADaPT). Its license remains in `vendor/AgentGym/LICENSE`. The client parser is reproduced with a parity test against the upstream function; the conversation prompt is read directly from the pinned source.
