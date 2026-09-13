import ast
import json
import re
from pathlib import Path
from dataclasses import dataclass
import pytest
from failure_probes.environment import TextCraft, parse_action, upstream_conversation, UPSTREAM


@pytest.fixture(scope="module")
def env():
    return TextCraft()


def test_reset_is_identical_and_inventory_is_fresh(env):
    task = env.freeze_task(10)
    first = env.reset(task)
    # Derive a gettable item from the task's own recipes rather than hardcoding
    # one: which items are craftable (and so NOT gettable) depends on recipe
    # load order, so a hardcoded name silently rots.
    item = _first_base_ingredient(env, task)
    assert item is not None, "no base ingredient to collect for this task"
    env.step(f"Thought: gather\nAction: get {item[0]} {item[1]}")
    assert env.env.inventory
    assert env.reset(task) == first
    assert env.env.inventory == {}


def test_real_environment_errors_and_success(env):
    # Under the pinned ascending load order, upstream's cycle breaking makes
    # gold_ingot craftable and gold_nugget a base item. Assert that first, so a
    # future ordering change fails here legibly instead of deep in a rollout.
    assert env.tree.is_craftable("minecraft:gold_ingot")
    assert not env.tree.is_craftable("minecraft:gold_nugget")
    task = env.freeze_task(10)
    task.update(goal="minecraft:gold_ingot", commands="craft 1 gold ingot using 9 gold nugget",
                initial_observation="Crafting commands:\ncraft 1 gold ingot using 9 gold nugget\n\nGoal: craft gold ingot.")
    env.reset(task)
    assert env.step("nonsense")["validity"] == "incorrectly_formatted"
    assert env.step("Action: craft 1 gold ingot using 9 gold nugget")["validity"] == "impossible"
    # `get` only yields base items, so the craftable goal cannot be collected
    # directly; the nuggets it is made from can be, because the cycle break left
    # them as a base item.
    assert env.step("Action: get 1 gold ingot")["validity"] == "impossible"
    assert env.step("Action: get 9 gold nugget")["validity"] == "executable"
    result = env.step("Action: craft 1 gold ingot using 9 gold nugget")
    assert result["reward"] == 1 and result["done"]


def test_client_parser_matches_actual_upstream():
    source = ast.parse((UPSTREAM / "agentenv/agentenv/envs/textcraft.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "TextCraftEnvClient")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    @dataclass
    class StepOutput:
        state: str
        reward: float
        done: bool
    namespace = {"re": re, "StepOutput": StepOutput}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "upstream_step", "exec"), namespace)
    class FakeClient:
        def _post(self, _path, data):
            self.action = data["action"]
            return {"observation": "result", "reward": 0, "done": False}
    for text in ["Action: get 1 gold ingot", "Action:\nget 2 oak_log!", "no action", "Action: get 1 gold ingot\nAction: inventory", "Thought: hi\nAction: inventory\n"]:
        fake = FakeClient()
        result = namespace["step"](fake, text)
        action, _, error = parse_action(text)
        if error:
            assert result.state == error
        else:
            assert fake.action == action
    assert upstream_conversation()[0]["role"] == "user"


def test_action_span_excludes_trailing_whitespace():
    text = "Thought: gather\nAction: get 1 gold ingot  \n"
    _, span, _ = parse_action(text)
    assert text[slice(*span)] == "get 1 gold ingot"


def _recipes_for_goal(env, task):
    for command in task["commands"].splitlines():
        match = re.match(r"craft (.*) using (.*)", command)
        if not match:
            continue
        recipe = env.env.extract_recipe(match.group(1), match.group(2))
        if recipe is not None and recipe.output_item.item_tag.name == task["goal"]:
            yield command, recipe


def _ingredient_name(ingredient):
    return ingredient.item_tag.name.replace("minecraft:", "").replace("_", " ")


def _first_base_ingredient(env, task):
    """(count, name) of an ingredient that can actually be collected."""
    for _command, recipe in _recipes_for_goal(env, task):
        for ingredient in recipe.input_items:
            if not env.tree.is_craftable(ingredient.item_tag.name):
                return ingredient.count, _ingredient_name(ingredient)
    return None


def test_debug_tasks_are_solvable_using_supplied_recipes(env):
    # Tasks are generated here rather than loaded from a frozen fixture: that is
    # sound only because recipe load order is now canonical, which is the point.
    # A task counts as solvable if ANY of its supplied recipes for the goal can
    # be completed - not merely whichever one load order happens to surface.
    solved = 0
    for index in range(10):
        task = env.freeze_task(index)
        env.reset(task)  # _recipes_for_goal needs env.env; do not rely on fixture state
        for command, recipe in _recipes_for_goal(env, task):
            env.reset(task)
            if all(env.step(f"Action: get {i.count} {_ingredient_name(i)}")["validity"] == "executable"
                   for i in recipe.input_items):
                result = env.step("Action: " + command)
                if result["reward"] == 1 and result["done"]:
                    solved += 1
                    break
    assert solved >= 3, f"only {solved} of the first 10 depth-1 tasks were solvable end to end"
