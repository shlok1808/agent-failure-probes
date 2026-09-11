import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def locate_action_tokens(tokenizer, generated_ids, span):
    """Return generated-token positions overlapping the raw Action span.

    Exact token reconstruction is checked. If the tokenizer cannot round-trip,
    fail rather than silently select an EOS/padding token or the wrong action.
    """
    if span is None:
        return []
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] != generated_ids:
        raise ValueError("Generated tokens did not round-trip for action alignment")
    start, end = span
    return [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]


class Actor:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model"], revision=config["model_revision"], dtype=self.dtype,
            attn_implementation="sdpa").to(self.device).eval()
        self.layer_number = config["layer_number"]
        if not 1 <= self.layer_number <= len(self.model.model.layers):
            raise ValueError("layer_number uses one-based decoder-block indexing")

    def generate(self, messages, seed, keep_logits=False):
        set_seed(seed)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        if ids.shape[1] + self.config["max_new_tokens"] > self.config["context_limit"]:
            raise ValueError("Context limit reached; no silent truncation permitted")
        started = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(
                ids, attention_mask=torch.ones_like(ids), do_sample=True,
                temperature=self.config["temperature"], top_p=self.config["top_p"],
                top_k=self.config["top_k"], repetition_penalty=1.0,
                max_new_tokens=self.config["max_new_tokens"],
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True, output_logits=keep_logits,
                output_scores=False)
        generated = out.sequences[0, ids.shape[1]:].tolist()
        logprobs = None
        if keep_logits:
            logprobs = [float(torch.log_softmax(logit[0].float(), dim=-1)[tok])
                        for logit, tok in zip(out.logits, generated)]
        special = set(self.tokenizer.all_special_ids)
        content = [i for i, tok in enumerate(generated) if tok not in special]
        if not content:
            raise ValueError("Model returned no content tokens")
        record = {"prompt": prompt, "prompt_token_ids": ids[0].tolist(),
                  "generated_token_ids": generated,
                  "raw_response": self.tokenizer.decode(generated[:content[-1] + 1], skip_special_tokens=False),
                  "response": self.tokenizer.decode(generated, skip_special_tokens=True),
                  "token_logprobs": logprobs, "last_content_token": content[-1],
                  "generation_seconds": time.monotonic() - started,
                  "hit_token_limit": generated[-1] not in special and len(generated) == self.config["max_new_tokens"]}
        del out
        return record

    def replay(self, record):
        ids = record["prompt_token_ids"] + record["generated_token_ids"]
        prompt_len = len(record["prompt_token_ids"])
        target = prompt_len + record["probe_generated_token_index"]
        captured = {}
        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["residual"] = hidden[0, target].detach().float().cpu().numpy().copy()
        handle = self.model.model.layers[self.layer_number - 1].register_forward_hook(capture)
        try:
            with torch.inference_mode():
                # Call the backbone, avoiding sequence-length x vocabulary logits.
                output = self.model.model(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
            normalized = output.last_hidden_state[0, target].float().cpu().numpy().copy()
        finally:
            handle.remove()
        if not np.isfinite(captured["residual"]).all():
            raise ValueError("Non-finite hidden state")
        return captured["residual"], normalized
