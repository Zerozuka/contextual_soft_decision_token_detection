"""Receiver: information-bit LLRs -> tokens -> reconstructed images."""

import time
from typing import Optional

import torch
from loguru import logger
from omegaconf import DictConfig

from .base import CommunicationMethod
from .emu3 import Emu3SoftDecoder, Emu3Wrapper


class Receiver:
    """Recover tokens from LLRs and reconstruct images with the VQ decoder.

    - ``receiver.use_llm=false``: hard-decision baseline (no LMM prior).
    - ``receiver.use_llm=true``: proposed LMM-based joint detection
      (:class:`Emu3SoftDecoder`, greedy).
    """

    def __init__(self, cfg: DictConfig, emu: Emu3Wrapper):
        self.cfg = cfg
        self.emu = emu
        if cfg.get("beam_width", 1) != 1:
            raise ValueError(
                "Only greedy decoding (receiver.beam_width=1) is supported. "
                "A beam-search variant that was not used in the paper is kept in "
                "other/emu3_legacy.py."
            )
        self._decoder: Optional[Emu3SoftDecoder] = None
        self._decoder_key: Optional[tuple] = None

    def __call__(self, transmitted: dict) -> tuple[torch.Tensor, dict, dict]:
        t_start = time.time()
        spatial_shape = transmitted["spatial_shape"]
        tokens_per_image = spatial_shape[0] * spatial_shape[1]
        llr = transmitted["llr"]

        if not self.cfg.get("use_llm", True):
            comm = CommunicationMethod(
                name="baseline",
                bits_per_token=transmitted["bits_per_token"],
                num_bits_per_symbol=self.cfg.get("num_bits_per_symbol", 2),
            )
            rx_bits, rx_tokens = comm.decode(llr)
        else:
            decoder_key = (spatial_shape, tokens_per_image)
            if self._decoder_key != decoder_key:
                self._decoder = Emu3SoftDecoder(
                    emu=self.emu,
                    tokens_per_image=tokens_per_image,
                    spatial_shape=spatial_shape,
                    num_bits_per_symbol=self.cfg.get("num_bits_per_symbol", 2),
                )
                self._decoder_key = decoder_key
            rx_bits, rx_tokens = self._decoder.decode(
                llr,
                temperature=self.cfg.get("temperature", 1.0),
                use_cache=self.cfg.get("use_cache", True),
            )
            self._decoder.reset_cache()
        elapsed_decode = time.time() - t_start

        # Offload the LMM before running the VQ decoder (no-op without the LMM)
        self.emu.clear_llm_memory()

        # VQ-decode one image at a time to save memory
        recon_list = []
        for i in range(len(rx_tokens) // tokens_per_image):
            tokens = rx_tokens[i * tokens_per_image : (i + 1) * tokens_per_image]
            recon_list.append(self.emu.decode(tokens, spatial_shape))
        recon_tensor = torch.cat(recon_list, dim=0)
        logger.debug(f"VQ decode of {len(recon_list)} images done")

        rx_times = {"decode": elapsed_decode, "total": time.time() - t_start}
        rx_data = {"rx_tokens": rx_tokens, "rx_bits": rx_bits}
        return recon_tensor, rx_times, rx_data
