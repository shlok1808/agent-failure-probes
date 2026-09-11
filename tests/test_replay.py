import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from failure_probes.actor import Actor, locate_action_tokens


def tiny_actor():
    torch.manual_seed(1)
    actor = Actor.__new__(Actor)
    actor.device = "cpu"
    actor.layer_number = 2
    actor.model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8)).eval()
    return actor


def test_replay_reads_selected_token_before_final_normalization():
    actor = tiny_actor()
    record = {"prompt_token_ids": [1, 2, 3], "generated_token_ids": [4, 5, 6], "probe_generated_token_index": 1}
    residual, normalized = actor.replay(record)
    assert residual.shape == normalized.shape == (16,)
    expected = actor.model.model.norm(torch.tensor(residual)).detach().numpy()
    np.testing.assert_allclose(expected, normalized, atol=1e-6)
    assert not np.allclose(residual, normalized)
    # Tokens AFTER the probed action site cannot change that site's activation.
    record["generated_token_ids"][-1] = 9
    future_changed, _ = actor.replay(record)
    np.testing.assert_array_equal(residual, future_changed)
    record["generated_token_ids"][1] = 10
    action_changed, _ = actor.replay(record)
    assert not np.allclose(residual, action_changed)


def test_action_alignment_does_not_select_eos():
    class CharacterTokenizer:
        def decode(self, ids, **kwargs):
            return "".join(chr(i) for i in ids)
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i+1) for i in range(len(text))]}
    text = "Action: get 1 wood<eos>"
    positions = locate_action_tokens(CharacterTokenizer(), [ord(c) for c in text], (8, 18))
    assert text[positions[-1]] == "d"


def test_noncanonical_generated_tokens_are_not_reencoded():
    class Tokenizer:
        pieces = {1: "Action: ", 2: "in", 3: "ventory", 4: "<eos>"}
        def decode(self, ids, **kwargs):
            return "".join(self.pieces[i] for i in ids)
        def __call__(self, text, **kwargs):
            # Canonical tokenizer would merge the two sampled action tokens.
            return {"input_ids": [1, 99, 4], "offset_mapping": [(0, 8), (8, 17), (17, 22)]}
    assert locate_action_tokens(Tokenizer(), [1, 2, 3, 4], (8, 17)) == [1, 2]
