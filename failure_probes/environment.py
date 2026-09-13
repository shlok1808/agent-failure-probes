"""Use unchanged upstream mechanics in process; no HTTP server needed."""
import ast
import contextlib
import hashlib
import importlib
import os
import re
import sys
import types
from copy import deepcopy
from .common import ROOT, digest

UPSTREAM = ROOT / "vendor/AgentGym"
COMMIT = "3ef9235d23e68e7c2920c5422ad957dc8ced5c6c"
PACKAGE = UPSTREAM / "agentenv-textcraft/agentenv_textcraft"


RECIPE_ORDER_POLICY = "sorted(os.listdir)"


class _CanonicalOS:
    """Only the two `os` attributes `_load_recipes` actually uses.

    Anything else (scandir, walk) raises AttributeError loudly rather than
    silently reintroducing filesystem order after an upstream bump.
    """

    path = os.path

    @staticmethod
    def listdir(directory):
        return sorted(os.listdir(directory))


@contextlib.contextmanager
def canonical_recipe_order(module):
    """Swap one module's `os` binding for the duration of tree construction.

    Upstream reads recipes with a bare `os.listdir` (crafting_tree.py:61), the
    only filesystem-order call site in the package. Directory order differs
    between APFS and ext4, and upstream breaks recipe cycles by whichever file
    loads first, so the order decides both `data_idx` -> goal and which items
    are craftable at all. Scoped to the single constructor call; `vendor/` must
    stay byte-identical because prepare() rejects a dirty submodule.
    """
    original = module.os
    module.os = _CanonicalOS
    try:
        yield
    finally:
        module.os = original


def upstream_classes():
    # Upstream __init__ starts a global server with a cwd-relative recipe path.
    # A private namespace loads the identical mechanics without that side effect.
    name = "_pinned_textcraft"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(PACKAGE)]
        sys.modules[name] = module
    tree_module = importlib.import_module(name + ".crafting_tree")
    env = importlib.import_module(name + ".environment").TextCraftEnv
    return tree_module.CraftingTree, env, tree_module


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
        tree_cls, self.env_cls, tree_module = upstream_classes()
        with canonical_recipe_order(tree_module):
            self.tree = tree_cls(minecraft_dir=str(PACKAGE))

    def _depth_ordered_pool(self):
        """Replicate upstream's goal selection exactly (environment.py:167-169).

        Indices must be the `data_idx` values upstream will actually resolve, so
        this uses the same generator and the same stable sort rather than
        re-deriving an order from item names.
        """
        pairs = list(self.tree.item_recipes_min_depth(1))
        return sorted(pairs, key=lambda x: x[1])

    def tasks_by_depth(self, depths):
        """data_idx values whose goal has one of `depths` as its min recipe depth."""
        depths = set(depths)
        return [i for i, (_item, depth) in enumerate(self._depth_ordered_pool()) if depth in depths]

    def fingerprint(self):
        """Two independent digests, so a cross-machine mismatch is diagnosable.

        A differing corpus hash means the recipe data or submodule drifted; a
        matching corpus hash with a differing tree hash means the load order or
        cycle resolution changed. Every set is serialised sorted, so both are
        independent of PYTHONHASHSEED and safe to pin in tests.
        """
        recipes = PACKAGE / "recipes"
        names = sorted(os.listdir(recipes))
        corpus = [[n, hashlib.sha256((recipes / n).read_bytes()).hexdigest()] for n in names]
        pool = self._depth_ordered_pool()
        histogram = {}
        for _item, depth in pool:
            histogram[str(depth)] = histogram.get(str(depth), 0) + 1
        tree_state = {
            # Insertion order is deliberately preserved: it is the thing being pinned.
            "itemid_recipes": [[str(k), [str(r) for r in v]] for k, v in self.tree.itemid_recipes.items()],
            "tag_recipes": [[str(k), [str(r) for r in v]] for k, v in self.tree.tag_recipes.items()],
            "itemid_set": sorted(str(i) for i in self.tree.itemid_set),
            "item_id_to_tag": sorted((str(k), str(v)) for k, v in self.tree.item_id_to_tag.items()),
            "tag_set": sorted(str(t) for t in self.tree.tag_set),
            "pool": [[str(getattr(item, "name", item)), depth] for item, depth in pool],
        }
        return {"policy": RECIPE_ORDER_POLICY, "recipe_file_count": len(names),
                "recipe_corpus_hash": digest(corpus), "crafting_tree_hash": digest(tree_state),
                "pool_size": len(pool), "depth_histogram": histogram}

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
