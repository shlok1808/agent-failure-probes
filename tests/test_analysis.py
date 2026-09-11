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
