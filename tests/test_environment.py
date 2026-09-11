import ast
import re
from dataclasses import dataclass
import pytest
from failure_probes.environment import TextCraft, parse_action, upstream_conversation, UPSTREAM


@pytest.fixture(scope="module")
def env():
    return TextCraft()


def test_reset_is_identical_and_inventory_is_fresh(env):
    task = env.freeze_task(10)
    first = env.reset(task)
    env.step("Thought: gather\nAction: get 1 gold ingot")
    assert env.env.inventory
    assert env.reset(task) == first
    assert env.env.inventory == {}


def test_real_environment_errors_and_success(env):
    task = env.freeze_task(10)
    task.update(goal="minecraft:gold_nugget", commands="craft 9 gold nugget using 1 gold ingot",
                initial_observation="Crafting commands:\ncraft 9 gold nugget using 1 gold ingot\n\nGoal: craft gold nugget.")
    env.reset(task)
    assert env.step("nonsense")["validity"] == "incorrectly_formatted"
    assert env.step("Action: craft 9 gold nugget using 1 gold ingot")["validity"] == "impossible"
    assert env.step("Action: get 1 gold ingot")["validity"] == "executable"
    result = env.step("Action: craft 9 gold nugget using 1 gold ingot")
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
