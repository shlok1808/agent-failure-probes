import numpy as np
from failure_probes.analysis import within_task, grouped_scores, bootstrap_within


def test_task_difficulty_alone_has_no_within_signal():
    result = within_task([1, 0, 1, 0], [.9, .9, .1, .1], ["hard", "hard", "easy", "easy"])
    assert result["pair_weighted_auc"] == .5


def test_perfect_reversed_and_no_mixed():
    assert within_task([1, 0], [.9, .1], ["a", "a"])["macro_auc"] == 1
    assert within_task([1, 0], [.1, .9], ["a", "a"])["macro_auc"] == 0
    assert within_task([1, 0], [.9, .1], ["a", "b"])["pair_weighted_auc"] is None


def test_pair_and_macro_weighting_differ():
    result = within_task([1, 0, 1, 1, 0, 0], [.9, .1, .1, .1, .9, .9], ["a", "a", "b", "b", "b", "b"])
    assert result["pair_weighted_auc"] == .2
    assert result["macro_auc"] == .5
    assert bootstrap_within(result["per_task"], samples=20, seed=1) is not None


def test_grouped_predictions_never_see_their_task():
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], 10)
    groups = np.repeat(np.arange(10), 2)
    x = rng.normal(size=(20, 4))
    scores, splits = grouped_scores(x, y, groups)
    assert np.isfinite(scores).all()
    tested = []
    for split in splits:
        assert not set(split["train_tasks"]) & set(split["test_tasks"])
        tested.extend(split["test"])
    assert sorted(tested) == list(range(20))


def test_one_class_is_not_fake_auc():
    scores, splits = grouped_scores(np.ones((4, 2)), np.zeros(4), np.array(["a", "a", "b", "b"]))
    assert np.isnan(scores).all() and not splits


def test_shard_slices_are_disjoint_and_cover_every_task():
    """Sharding must partition the tasks exactly: no episode run twice, none skipped.

    The seed is generation_seed + data_idx*100000 + attempt*100 + round, so it
    does not depend on execution order - N workers over disjoint slices produce
    the same episodes as one worker. That only holds if the slices partition.
    """
    tasks = [f"textcraft_{i}" for i in range(125)]
    for count in [1, 2, 4, 6, 8, 125, 200]:
        slices = [tasks[i::count] for i in range(count)]
        flat = [t for s in slices for t in s]
        assert sorted(flat) == sorted(tasks), f"{count} shards do not cover every task"
        assert len(flat) == len(set(flat)), f"{count} shards overlap"


def test_shard_striding_spreads_adjacent_tasks():
    """Stride, not blocks. Slow tasks cluster alphabetically - the plural-trap
    *_planks goals sit near each other - so block slicing would pile them onto
    one worker that then runs long after the others finished. Striding sends
    every run of consecutive tasks to distinct workers."""
    tasks = list(range(125))
    for count in [2, 4, 6, 8]:
        owner = {t: i for i in range(count) for t in tasks[i::count]}
        for start in range(0, 125 - count):
            window = [owner[t] for t in tasks[start:start + count]]
            assert len(set(window)) == count, f"{count} shards collide on {window}"
