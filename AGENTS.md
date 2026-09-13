# Handoff notes for coding agents

Read this before touching anything. Written 2026-09-11, after the debug run and
before the stage-2 collection. It records the things that are not obvious from
the code and that have already cost us one invalid run.

## Where the project stands

- `runs/debug-002` completed and is **inconclusive by construction**: only one of
  five tasks had both a success and a failure, so every within-task number came
  from one fold with **2 comparison pairs**. All four feature sets scored an
  identical 0.25. That is a sample-size artefact, not a broken probe. Do not
  treat 0.25 / 0.17 as a result, and do not "fix" the probe in response to it.
- Stage 2 has **not been collected yet**. The code has landed, tests pass on both
  platforms, and cross-platform determinism is verified - see below. What is
  unexercised is the collection path itself.

## The bug that shaped everything

TextCraft task numbers do not identify tasks.

Upstream picks a goal with `sorted(item_depth_list, key=depth)[data_idx]`
(`vendor/AgentGym/.../environment.py:167-169`), and that list is built from an
unsorted `os.listdir` over 860 recipe files (`.../crafting_tree.py:61`). The sort
only orders by depth, so within a depth tier the order is pure filesystem order.

Consequences, all verified empirically:

- `textcraft_0` is `acacia_planks` under ascending order, `yellow_dye` under
  descending, and was `dropper` on the original macOS run.
- It is **not just a reordering**. Upstream breaks recipe cycles by whichever
  file loads first, so load order changes *which items are craftable*:
  ascending makes `gold_ingot` craftable and `gold_nugget` a base item;
  descending flips all three of gold / iron / honey.
- The depth histogram itself moves: `{1:125, 2:291, 3:117, 4:11}` ascending vs
  `{1:147, 2:274, 3:112, 4:11}` descending. **"125 depth-1 tasks" is a
  consequence of choosing ascending sort, not a property of TextCraft.**

So macOS and Linux ran genuinely different experiments. This is also why CI went
red at `c28730f` with `assert 'impossible' == 'executable'`.

**Fix:** `canonical_recipe_order()` in `failure_probes/environment.py` swaps the
`crafting_tree` module's `os` binding for a whitelist shim during tree
construction only. `vendor/` must stay byte-identical — `prepare()` rejects a
dirty submodule, even an untracked file. Do not "simplify" this by editing
vendor, subclassing `_load_recipes` (that forks 118 lines of upstream), or
re-sorting after construction (**incorrect** — skipped recipes were never added,
so no re-sort recovers them).

## Rules that are easy to violate

1. **`PYTHONHASHSEED=0` on every invocation.** Upstream builds prompts from
   `random.sample(list(<set of str>))`. `prepare()` and `collect()` now refuse to
   run without it, and CI sets it.
2. **No edits to `failure_probes/*.py` once collection starts.**
   `pipeline.py` hashes every module into `runtime.json` and refuses to resume a
   run if any hash changed. A one-character fix mid-run means restarting all 1000
   episodes. (This guard was deliberately left strict; loosen it only with the
   owner's say-so.)
3. **Run `prepare` on the machine that will collect.** The manifest records
   `platform` and `packages`; preparing on the Mac and collecting on Lambda
   writes a manifest describing the wrong machine.
4. **Debug data must never become confirmatory data.** `prepare` rejects a
   stage-2 config aimed at a `runs/debug*` directory.
5. **Do not add a `--task-ids` CLI flag.** Hand-picked task lists are a
   selection-bias footgun; selection lives in config so it is hashed into the
   manifest.

## What changed in this pass

| Area | Change |
|---|---|
| `environment.py` | `canonical_recipe_order()` shim; `fingerprint()` (two digests); `tasks_by_depth()` |
| `pipeline.py` | stage allowlist; `require_deterministic_hashing()`; config-driven task selection; fingerprint into manifest + checked in `collect`/`audit`; retry-and-continue instead of re-raise; per-episode progress + `progress.jsonl`; streaming `audit`/`extract` |
| `analysis.py` | **aliasing bug fixed** (`mask &= ...` mutated `valid` in place across models); streaming; refuses incomplete runs unless `--allow-incomplete`; stage-aware prose |
| `common.py` | `iter_episodes()`, `episode_paths()`, `append_jsonl()` |
| `cli.py` | `prepare --dry-run`, `analyze --allow-incomplete` |
| `configs/` | new `stage2.json` (125 × 8); `task_selection` + `max_episode_retries` added to `debug.json` |
| `tests/` | new `test_ordering.py` (goldens); two env tests repaired; `fixtures/debug_tasks.json` deleted |
| CI | `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1` |

Why `generation_seed` is `20260912` in stage 2, not `20260911`: the seed is
`generation_seed + data_idx*100000 + attempt*100 + round`, so reusing the debug
value would replay identical rollouts for overlapping indices.

## Determinism: VERIFIED 2026-09-13 (re-verified after the audit fixes)

Mac and Lambda dry-runs agree. This closes the platform bug.

| Field | Result |
|---|---|
| `crafting_tree_hash` | `8ec9aad5989b4afe9a01c5d69c9648b23ef59dbd1517aa8aa647ee0c5db2367c` - identical |
| `recipe_corpus_hash` | `2b11d08046dc9f217732a6d301b433dfcebbae54af1675552ec55d6d3efa9da8` - identical |
| `selection_digest` | `a900ecc5a9d33377e4e8b8e6eb63c977b802a65b8c49c3a7eb9514c081ec2fa9` - identical |
| All 125 task entries | 0 mismatches (`data_idx`, `goal`, `recipe_depth`, `task_hash`) |
| `manifest_hash` | **differs, and that is correct** |

`manifest_hash` is computed over the whole manifest including `platform`,
`python` and `packages`, which legitimately differ between machines. A plain
`diff` of the two dry-run files therefore reports one line. Compare
`recipe_order`, `selection_digest` and the task table - not `manifest_hash`.

## Lambda environment (as provisioned)

A100-SXM4-40GB, us-east-1, 472G free. The box shipped with **Python 3.10 only**,
below this project's `requires-python >=3.11`, so 3.13.15 was installed with
`uv` to match the Mac's 3.13.7 minor version. Repo delivered by `rsync` (not a
clone) because it is private; `.git` and the submodule metadata came across, so
`prepare`'s pinned/clean submodule check passes.

```bash
export PATH="$HOME/.local/bin:$PATH"
cd ~/agent-failure-probes && source .venv/bin/activate && export PYTHONHASHSEED=0
```

**Package drift from `requirements-tested.txt`** - torch 2.14.0+cu130 (vs 2.13.0),
transformers 5.17.0 (vs 5.12.1), numpy 2.5.3, scikit-learn 1.9.1. Proven not to
affect task identity, but it is a real difference from the debug run's
environment and will be recorded in the stage-2 manifest. 21 tests pass there.

HF Hub is unauthenticated (slower downloads, possible rate limits). Set
`HF_TOKEN` if the model download stalls.

## What has NOT been verified

Tests now pass on both platforms (21 on Mac, 21 on Lambda) and CI is green, so
the items below are no longer untested - the determinism check above is done.
**What remains unexecuted is the collection itself**: no `prepare` into a real
run directory, no `collect`, no `extract`, no `analyze` at scale. Highest-risk
remaining items, in order:

1. **The `collect` retry loop has never executed.** It parses and passes tests
   that never reach it. The retry path, the `errors/*.retry{k}.json` writes and
   the `FATAL_ERRORS` allowlist are all unexercised. The first real fault is the
   first test of that code.
2. **`analyze` at scale is unexercised.** The streaming rewrite, the
   incomplete-run refusal and the stage-aware prose have never run on a real
   analysis - not even on the debug shape since the rewrite.
3. **`extract` on CUDA is unexercised.** The layer-28 hook has only ever run on
   MPS, and `extract` refuses to run if device/dtype differ from collection.
4. **Mixed-outcome yield is unknown** until roughly 2 h into collection. See the
   risk section - this is the one that decides whether the run is worth anything.

## Runbook

Mac repo lives at `~/Documents/ChatGPT/interp psu/agent-failure-probes`.

### 0. Lambda setup (Lambda, once)

```bash
git clone --recurse-submodules https://github.com/shlok1808/agent-failure-probes.git
cd agent-failure-probes
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0     # so "cuda" in runtime.json is unambiguous
```

### 1. Tests (BOTH machines - green on the Mac alone proves nothing)

```bash
PYTHONHASHSEED=0 pytest -q
```

### 1b. macOS only: the console script may not work

**Root cause, confirmed in the stdlib:** Python 3.13's `site.addpackage`
(`site.py:177-179`) silently skips any `.pth` file carrying the macOS
`UF_HIDDEN` flag. Every file in this venv's `site-packages` had that flag, so
the editable install's `.pth` never executed, the import hook was never
registered, and `failure_probes` was invisible - with no error anywhere.

```bash
chflags -R nohidden .venv     # RECURS - the flag came back on its own, most
                              # likely iCloud syncing ~/Documents. Re-apply
                              # whenever imports break, or just use python -m.
```

**Prefer `python -m` on the Mac**: `python -m failure_probes.cli` and
`python -m pytest` bypass the `.pth` entirely and cannot hit this. The flag has
been observed returning after being cleared, so the console script is not
trustworthy there. Lambda is unaffected.

**`pytest` masks this** - it imports from the working directory, so green tests
do not prove the console script works. The dry-run below is the first command
that exercises it. Linux is unaffected (no `UF_HIDDEN`), so this cannot change
any digest; it only decides whether the command runs at all.

### 2. Determinism check (BOTH machines, then diff)

```bash
# Mac
PYTHONHASHSEED=0 failure-probes prepare --config configs/stage2.json --run /tmp/dry --dry-run > /tmp/dry-mac.json
# Lambda
PYTHONHASHSEED=0 failure-probes prepare --config configs/stage2.json --run /tmp/dry --dry-run > /tmp/dry-lambda.json
# either, after copying one across
diff /tmp/dry-mac.json /tmp/dry-lambda.json && echo MATCH
```

Must print `MATCH` before collecting. If `crafting_tree_hash` differs →
load-order problem. If it matches but `task_hash`es differ → `PYTHONHASHSEED`
or Python-version mismatch in `commands`.

The dry-run prints only `recipe_order`, `manifest_hash`, `selection_digest` and
the task table - deliberately not `packages`. The Mac venv inherits system
site-packages (torch, transformers, scikit-learn resolve to system Python), so
package versions legitimately differ between machines and land in the manifest.
Only `recipe_order` and the `task_hash` values must match.

### 3. Stage 2 (LAMBDA only) - PARALLEL

One worker leaves the A100 at ~33% (batch-1 decode is bandwidth-bound), giving a
~10h run. Six workers over disjoint task slices bring it to roughly 1.5-2h.

**Why this is safe:** the per-round seed is
`generation_seed + data_idx*100000 + attempt*100 + round` - it depends only on
which task/attempt/round it is, never on execution order. N workers over
disjoint slices therefore produce byte-identical episodes to one worker.
**Batching inside a generate() call would NOT be safe**, because
`actor.generate` calls `set_seed` per call; do not try it.

Slices stride (`tasks[i::n]`), not block, because the slow plural-trap tasks
cluster alphabetically and blocks would leave one worker running long.

### 3b. Single-worker form (slower, still correct)

```bash
tmux new -s stage2                 # detach Ctrl+B D, reattach: tmux attach -t stage2
cd agent-failure-probes && source .venv/bin/activate
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0

failure-probes prepare --config configs/stage2.json --run runs/stage2-001
failure-probes collect --run runs/stage2-001
failure-probes audit   --run runs/stage2-001
failure-probes extract --run runs/stage2-001
failure-probes analyze --run runs/stage2-001
```

Monitor from a second shell: `tail -f runs/stage2-001/progress.jsonl`

Resume after any interruption - same command, finished episodes are skipped:
`failure-probes collect --run runs/stage2-001`

**Do not prepare a separate small pilot.** `collect` refuses episodes whose
`manifest_hash` differs, so pilot episodes can never join the real run. Instead
prepare the full manifest, start `collect`, and pause after ~2 h — tasks iterate
outer and attempts inner, so you get complete 8-attempt tasks to estimate the
mixed-outcome yield, then resume with zero waste.

## The risk worth watching

The run's value depends on **mixed-outcome tasks** (both a success and a failure
among 8 attempts). Debug showed 73% success. At per-attempt success *p*: p=0.85 →
~91 mixed tasks; p=0.95 → ~42; p=0.98 → ~19 and the analysis is starved. Check
this at the 2 h pause. If *p* is above ~0.93, argue for adding depth-2 tasks
(`"recipe_depths": [1, 2]`) **before** `prepare` — it is a one-line config change
then, and impossible afterwards without a new run. Depth-2 is currently scoped as
a follow-up.

Target for context: Ruan reports first-round AUC **0.81** for this cell.
