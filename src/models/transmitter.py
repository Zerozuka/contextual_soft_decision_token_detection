"""Transmitter: image tensor -> Emu3 VQ tokens -> bits."""

import time

import numpy as np
import torch
from omegaconf import DictConfig

from .base import tokens_to_bits
from .emu3 import Emu3Wrapper


class Transmitter:
    """Tokenize preprocessed images with the Emu3 VQ tokenizer and map tokens to bits.

    Images are expected to be resized / cropped by the dataset transform.
    """

    def __init__(self, cfg: DictConfig, emu: Emu3Wrapper):
        self.cfg = cfg
        self.emu = emu

    def __call__(self, img_tensor: torch.Tensor) -> tuple[dict, dict]:
        t_start = time.time()
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        indices, spatial_shape = self.emu.encode(img_tensor)
        elapsed_encode = time.time() - t_start

        tokens_flat = indices.reshape(-1).astype(np.int64)
        transmitted = {
            "tx_tokens": tokens_flat,
            "tx_bits": tokens_to_bits(tokens_flat, self.emu.bits_per_token),
            "spatial_shape": spatial_shape,
            "bits_per_token": self.emu.bits_per_token,
        }
        tx_times = {"encode": elapsed_encode, "total": time.time() - t_start}
        return transmitted, tx_times
