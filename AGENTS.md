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
- Stage 2 has **not been collected yet**. The code for it has just landed and is
  **untested** — see "What has not been verified".

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

## What has NOT been verified

**Nothing here has been executed.** The owner asked to run everything
themselves. No `pytest`, no smoke run, no `prepare`. Treat every change as
untested. Highest-risk items, in order:

1. `test_debug_tasks_are_solvable_using_supplied_recipes` — rewritten to generate
   tasks and require ≥3 of the first 10 to be solvable end to end. The threshold
   is a guess; if it fails, check whether these goals need craftable (therefore
   un-`get`table) ingredients before assuming the environment is broken.
2. `test_real_environment_errors_and_success` — inverted to gold_ingot/gold_nugget.
   Correct per the craftability check above, but unrun.
3. The `collect` retry loop was re-indented by hand. It parses; it has not run.
4. `fingerprint()` reads `itemid_recipes` / `tag_recipes` / `tag_set` directly —
   attribute names confirmed present, serialisation unrun.

## Current git state (as of handoff)

All changes below are **uncommitted, on `main`**, and the repo has no other
branch. Nothing has been pushed. Suggested first move:

```bash
git checkout -b stage2-determinism
git add -A && git commit
```

Deleted file to expect in the diff: `tests/fixtures/debug_tasks.json`.
New files: `AGENTS.md`, `configs/stage2.json`, `tests/test_ordering.py`.

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

The editable install's `.pth` and finder in `.venv/lib/python3.13/site-packages/`
carry the macOS hidden flag (visible as `hidden` in `ls -lO`), and
`.venv/bin/failure-probes` was observed failing with `ModuleNotFoundError`.
Clearing the flag fixed it:

```bash
chflags -R nohidden .venv
```

Always-works fallback: use `python -m failure_probes.cli` wherever the runbook
says `failure-probes`.

**`pytest` masks this** - it imports from the working directory, so green tests
do not prove the console script works. The dry-run below is the first command
that actually exercises it. Linux/Lambda is unaffected, so this cannot change
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

### 3. Stage 2 (LAMBDA only)

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
