"""Physical layer: channel coding, modulation, AWGN and demapping."""

import io
import time

import sionna as sn
import tensorflow as tf
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from omegaconf import DictConfig

from .base import CommunicationMethod, LDPCCommunicationMethod
from .emu3 import Emu3Wrapper


class Channel:
    """Encode -> QAM map -> AWGN -> APP demap (-> LDPC BP decode).

    ``__call__`` takes the dict produced by :class:`Transmitter` and returns a
    copy with an ``llr`` entry holding the information-bit LLRs (already
    BP-decoded when LDPC is enabled), plus the transmitted size in KB.
    """

    def __init__(self, cfg: DictConfig, emu: Emu3Wrapper):
        self.cfg = cfg
        if cfg.get("use_ldpc", False):
            self.comm = LDPCCommunicationMethod(
                name="channel_ldpc",
                bits_per_token=emu.bits_per_token,
                num_bits_per_symbol=cfg.get("num_bits_per_symbol", 2),
                n=cfg.get("ldpc_n", 2048),
                coderate=cfg.get("ldpc_coderate", 0.5),
                num_iter=cfg.get("ldpc_num_iter", 20),
            )
            logger.info(
                f"LDPC: k={self.comm.k}, n={self.comm.n}, "
                f"iterations={cfg.get('ldpc_num_iter', 20)}"
            )
        else:
            self.comm = CommunicationMethod(
                name="channel",
                bits_per_token=emu.bits_per_token,
                num_bits_per_symbol=cfg.get("num_bits_per_symbol", 2),
            )
        self._awgn = sn.channel.AWGN()

    @staticmethod
    def calculate_size_KB(image_batch: torch.Tensor) -> float:
        """Approximate PNG size [KB] of the batch (first image size x batch size)."""
        imgs = image_batch if image_batch.dim() == 4 else image_batch.unsqueeze(0)
        buf = io.BytesIO()
        TF.to_pil_image(imgs[0].cpu()).save(buf, format="PNG")
        return buf.tell() / 1024.0 * imgs.shape[0]

    def __call__(self, transmitted: dict) -> tuple[dict, dict]:
        tx_bits = transmitted["tx_bits"]
        n_info = int(len(tx_bits))

        # channel.snr is Eb/N0 [dB]
        no = sn.utils.ebnodb2no(
            self.cfg.snr, self.comm.num_bits_per_symbol, self.comm.coderate
        )

        t0 = time.time()
        coded_bits = self.comm.encode(tx_bits)  # [1, N_coded]
        x = self.comm.mapper(coded_bits)        # [1, N_symbols]
        y = self._awgn([x, no])
        llr = self.comm.demapper([y, no])       # [1, N_coded]
        if isinstance(self.comm, LDPCCommunicationMethod):
            llr = self.comm.ldpc_decode(llr, n_info)  # [N_info]
        elapsed = time.time() - t0

        logger.debug(
            f"LLR |mean|={tf.reduce_mean(tf.abs(llr)).numpy():.3f}, "
            f"|max|={tf.reduce_max(tf.abs(llr)).numpy():.3f}"
        )

        bits_tx = int(tf.size(x)) * self.comm.num_bits_per_symbol
        trans_sizes = {"total": bits_tx / 8 / 1024}

        transmitted = dict(transmitted)
        transmitted["llr"] = llr
        transmitted["channel_elapsed"] = elapsed
        return transmitted, trans_sizes
