"""Pin the recipe load order.

Upstream reads recipes with an unsorted `os.listdir`, so directory order decided
both which goal a `data_idx` resolves to and - through cycle breaking - which
items are craftable at all. macOS and Linux therefore ran different experiments.
These goldens are the committed cross-platform assertion; they are independent of
PYTHONHASHSEED because every set is serialised sorted.
"""
import os
import pytest
from failure_probes.environment import TextCraft, _CanonicalOS, PACKAGE

POOL_SIZE = 544
DEPTH_HISTOGRAM = {"1": 125, "2": 291, "3": 117, "4": 11}
GOLDEN_INDEX_TO_GOAL = {
    0: "minecraft:acacia_planks",
    1: "minecraft:acacia_wood",
    2: "minecraft:beacon",
    10: "minecraft:blue_dye",
    124: "minecraft:yellow_dye",
    125: "minecraft:acacia_boat",  # first depth-2 item
}


@pytest.fixture(scope="module")
def env():
    return TextCraft()


def test_index_to_goal_is_pinned(env):
    """The regression test the platform bug needed. A reordering moves these."""
    for index, goal in GOLDEN_INDEX_TO_GOAL.items():
        assert env.freeze_task(index)["goal"] == goal, f"data_idx {index} moved"


def test_depth_one_pool_is_the_stage2_task_set(env):
    indices = env.tasks_by_depth([1])
    assert indices == list(range(125))
    goals = [env.freeze_task(i)["goal"] for i in indices[:20]]
    assert len(set(goals)) == len(goals)


def test_pool_size_and_depth_histogram(env):
    fingerprint = env.fingerprint()
    assert fingerprint["pool_size"] == POOL_SIZE
    assert fingerprint["depth_histogram"] == DEPTH_HISTOGRAM


def test_cycle_resolution_is_pinned(env):
    """Ascending order is arbitrary but pinned; it flips these three pairs.

    Under the reverse order gold_nugget/iron_nugget/honey_bottle become the
    craftable side instead, and 22 items move into depth 1.
    """
    assert env.tree.is_craftable("minecraft:gold_ingot")
    assert not env.tree.is_craftable("minecraft:gold_nugget")
    assert env.tree.is_craftable("minecraft:iron_ingot")
    assert not env.tree.is_craftable("minecraft:iron_nugget")
    assert env.tree.is_craftable("minecraft:honey_block")
    assert not env.tree.is_craftable("minecraft:honey_bottle")


def test_fingerprint_is_stable_across_construction(env):
    assert TextCraft().fingerprint() == env.fingerprint()


def test_listdir_shim_is_scoped_to_construction(env):
    """The swap must not leak: it is restored in a finally block."""
    import _pinned_textcraft.crafting_tree as crafting_tree
    assert crafting_tree.os is os
    assert os.listdir is not _CanonicalOS.listdir


def test_shim_rejects_unexpected_os_attributes():
    """An upstream bump to scandir/walk must fail loudly, not silently reorder."""
    for attribute in ["scandir", "walk", "listdir_unsorted"]:
        if attribute == "listdir_unsorted" or not hasattr(os, attribute):
            continue
        with pytest.raises(AttributeError):
            getattr(_CanonicalOS, attribute)


def test_recipe_filenames_are_ascii():
    """Makes sorted() locale-independent."""
    for name in os.listdir(PACKAGE / "recipes"):
        assert name.isascii(), name
