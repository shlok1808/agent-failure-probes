"""Use unchanged upstream mechanics in process; no HTTP server needed."""
import ast
import importlib
import re
import sys
import types
from copy import deepcopy
from .common import ROOT, digest

UPSTREAM = ROOT / "vendor/AgentGym"
COMMIT = "3ef9235d23e68e7c2920c5422ad957dc8ced5c6c"
PACKAGE = UPSTREAM / "agentenv-textcraft/agentenv_textcraft"


def upstream_classes():
    # Upstream __init__ starts a global server with a cwd-relative recipe path.
    # A private namespace loads the identical mechanics without that side effect.
    name = "_pinned_textcraft"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(PACKAGE)]
        sys.modules[name] = module
    tree = importlib.import_module(name + ".crafting_tree").CraftingTree
    env = importlib.import_module(name + ".environment").TextCraftEnv
    return tree, env


def upstream_conversation():
    path = UPSTREAM / "agentenv/agentenv/envs/textcraft.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TextCraftEnvClient")
    assignment = next(n for n in cls.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "conversation_start" for t in n.targets))
    records = [ast.literal_eval(call.args[0]) for call in assignment.value.elts]
    return [{"role": "user" if r["from"] == "human" else "assistant", "content": r["value"]} for r in records]


def parse_action(response):
    """Match AgentGym's client extraction/normalization, preserving raw response."""
    matches = list(re.finditer(r"Action:\s*(.*?)(?=\n|$)", response, re.DOTALL))
    if len(matches) > 1:
        return "", None, "Error: Only one 'Action' is allowed per response. Please adjust your response."
    if not matches:
        return "", None, None
    match = matches[-1]
    action = " ".join(re.sub(r"[^A-Za-z0-9, ]+", "", match.group(1)).split()).strip()
    end = match.start(1) + len(match.group(1).rstrip())
    span = (match.start(1), end) if end > match.start(1) else None
    return action, span, None


def validity(observation, parse_error=None):
    if parse_error or observation.startswith(("Could not execute", "Could not extract recipe", "Wrong item format")):
        return "incorrectly_formatted"
    if observation.startswith("Could not find"):
        return "impossible"
    if observation.startswith(("Got ", "Crafted ", "Inventory:")):
        return "executable"
    return "unclassified"


class TextCraft:
    def __init__(self):
        tree_cls, self.env_cls = upstream_classes()
        self.tree = tree_cls(minecraft_dir=str(PACKAGE))

    def freeze_task(self, index, seed=42):
        env = self.env_cls(self.tree, None, None)
        observation, _ = env.reset(seed=seed, data_idx=index)
        task = {"task_id": f"textcraft_{index}", "data_idx": index, "environment_seed": seed,
                "goal": env.goal, "commands": env.commands, "initial_observation": observation,
                "recipe_depth": self.tree.min_depth.get(env.goal)}
        task["task_hash"] = digest(task)
        return task

    def reset(self, task):
        self.env = self.env_cls(self.tree, task["commands"], task["goal"])
        observation, _ = self.env.reset(commands=task["commands"], goal=task["goal"])
        if observation != task["initial_observation"]:
            raise ValueError("Frozen task reset changed its prompt")
        return observation

    def step(self, response):
        action, span, error = parse_action(response)
        before = deepcopy(self.env.inventory)
        if error:
            observation, reward, done = error, 0, False
        else:
            observation, reward, done, _, _ = self.env.step(action)
        return {"action": action, "action_char_span": span, "observation": observation,
                "reward": reward, "done": done, "validity": validity(observation, error),
                "inventory_before": before, "inventory_after": deepcopy(self.env.inventory)}
