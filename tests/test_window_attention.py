"""Unit test of the sliding-window attention in the vendored Emu3ForCausalLM.

Runs on CPU with a tiny randomly initialized 1-layer model (no download needed):

    python -m unittest discover tests

With a single layer the receptive field equals the attention window, so the
logits at the last position must
  - not change when image tokens before the window are changed,
  - change when a token inside the window is changed,
  - change when the prefix (before the start-of-image token) is changed,
both for a full forward pass and for incremental decoding with the KV cache
(as used by Emu3SoftDecoder).
"""

import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from models.third_party.configuration_emu3 import Emu3Config  # noqa: E402
from models.third_party.modeling_emu3 import Emu3ForCausalLM  # noqa: E402

SOT = 151851  # start-of-image token ID


def build_model():
    torch.manual_seed(0)
    cfg = Emu3Config(
        vocab_size=151860, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
        pad_token_id=0, attn_implementation="eager",
    )
    return Emu3ForCausalLM(cfg).eval()


def build_ids():
    g = torch.Generator().manual_seed(1)
    prefix = torch.randint(1, 1000, (5,), generator=g)
    img = torch.randint(1000, 30000, (200,), generator=g)
    return torch.cat([prefix, torch.tensor([SOT]), img]).unsqueeze(0)


def logits_last(model, ids, window, use_cache):
    model.model.window_size = window
    with torch.no_grad():
        if not use_cache:
            return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]
        # Prefill prefix + SOT, then feed one token at a time
        p = (ids[0] == SOT).nonzero()[0].item() + 1
        out = model(input_ids=ids[:, :p], attention_mask=torch.ones(1, p, dtype=torch.long), use_cache=True)
        past = out.past_key_values
        for t in range(p, ids.shape[1]):
            out = model(
                input_ids=ids[:, t : t + 1],
                attention_mask=torch.ones(1, t + 1, dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
        return out.logits[0, -1]


class WindowAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()
        cls.ids = build_ids()

    def _delta(self, other, window, use_cache, base):
        return (logits_last(self.model, other, window, use_cache) - base).abs().max().item()

    def test_window_mask(self):
        ids = self.ids
        n = ids.shape[1]
        first_img = (ids[0] == SOT).nonzero()[0].item() + 1
        for window in (10, 64, 128):
            for use_cache in (False, True):
                with self.subTest(window=window, use_cache=use_cache):
                    effective_window = window + window // 64
                    start = (n - 1) - effective_window + 1  # first visible position for the last query
                    base = logits_last(self.model, ids, window, use_cache)

                    outside = ids.clone()
                    outside[0, first_img:start] = torch.randint(1000, 30000, (start - first_img,))
                    inside = ids.clone()
                    inside[0, start + 1] = (inside[0, start + 1] + 1) % 30000 + 1000
                    prefix = ids.clone()
                    prefix[0, 0] = (prefix[0, 0] + 1) % 1000 + 1

                    self.assertLess(self._delta(outside, window, use_cache, base), 1e-5)
                    self.assertGreater(self._delta(inside, window, use_cache, base), 1e-4)
                    self.assertGreater(self._delta(prefix, window, use_cache, base), 1e-4)

    def test_large_window_sees_everything(self):
        base = logits_last(self.model, self.ids, 4096, False)
        far = self.ids.clone()
        far[0, 6] = (far[0, 6] + 1) % 30000 + 1000
        self.assertGreater(self._delta(far, 4096, False, base), 1e-4)


if __name__ == "__main__":
    unittest.main()
