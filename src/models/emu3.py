"""Emu3 components.

- :class:`Emu3Wrapper`: loads the Emu3 VQ tokenizer and, optionally, the Emu3-Gen LMM.
- :class:`Emu3SoftDecoder`: LMM-based joint semantic-channel token detection
  (greedy), fusing the token likelihood from the physical-layer LLRs with the
  LMM logits.
"""

import gc
import time
from typing import Tuple

import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from .base import CommunicationMethod, tokens_to_bits
from .third_party.modeling_emu3 import Emu3ForCausalLM
from .third_party.processing_emu3 import Emu3Processor


def _free_cache():
    gc.collect()
    gc.collect()  # reference cycles are not always collected in a single pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Emu3Wrapper:
    """Holds the Emu3 VQ tokenizer and (optionally) the Emu3-Gen LMM.

    - ``encode(images)``: [B, C, H, W] in [0, 1] -> VQ indices [B, H', W']
    - ``decode(tokens, spatial_shape)``: VQ indices -> images [B, C, H, W] in [0, 1]
    - ``prepare_prefix_prompt(H, W, batch_size)``: prefix token IDs for the LMM

    Models are kept on the CPU and moved to the accelerator only while in use,
    so that the LMM and the VQ tokenizer fit on a single GPU. Without the LMM
    (``use_llm=False``) the VQ tokenizer stays on the accelerator.
    """

    NUM_EMBED = 32768  # Emu3 VQ codebook size (2**15)

    def __init__(self, cfg, use_llm: bool = True):
        """
        Args:
            cfg: ``emu3`` config section with ``emu_hub``, ``vq_hub`` and ``window_size``.
            use_llm: if False, only the VQ tokenizer is loaded (hard-decision baseline).
        """
        self.cfg = cfg
        self.use_llm = use_llm

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.bfloat16

        self.num_embed = self.NUM_EMBED
        self.bits_per_token = int(np.log2(self.num_embed))
        self.window_size = cfg.window_size

        t0 = time.time()
        self.image_processor = AutoImageProcessor.from_pretrained(cfg.vq_hub, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(
            cfg.vq_hub, device_map="cpu", trust_remote_code=True, torch_dtype=self.dtype
        ).eval()
        logger.info(f"Loaded VQ tokenizer {cfg.vq_hub} ({time.time() - t0:.1f}s)")

        if use_llm:
            t0 = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.emu_hub, trust_remote_code=True, padding_side="left"
            )
            # The processor provides the visual-token template used for the VQ -> LMM vocab mapping
            self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)
            # Vendored Emu3ForCausalLM with sliding-window attention over image tokens
            self.model = Emu3ForCausalLM.from_pretrained(
                cfg.emu_hub,
                device_map="cpu",
                torch_dtype=self.dtype,
                trust_remote_code=False,
                attn_implementation="eager",
            ).eval()
            self.model.model.window_size = self.window_size
            self.eol_token_id = self.tokenizer.eos_token_id
            logger.info(
                f"Loaded LMM {cfg.emu_hub} ({time.time() - t0:.1f}s), "
                f"window_size={self.window_size}"
            )
        else:
            self.tokenizer = None
            self.processor = None
            self.model = None
            self.eol_token_id = None
            self.image_tokenizer.to(self._vq_device())

    @staticmethod
    def _vq_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # VQ encode / decode
    # ------------------------------------------------------------------

    def encode(self, images_tensor: torch.Tensor) -> Tuple[np.ndarray, Tuple[int, int]]:
        """[B, C, H, W] in [0, 1] -> (VQ indices [B, H', W'], (H', W'))."""
        pil_images = [TF.to_pil_image(img.cpu()) for img in images_tensor]
        device = self._vq_device()
        if self.use_llm:
            self.image_tokenizer.to(device)
            _sync()
        try:
            with torch.no_grad():
                inputs = self.image_processor(pil_images, do_resize=False, return_tensors="pt")[
                    "pixel_values"
                ].to(device, dtype=self.image_tokenizer.dtype)
                image_codes = self.image_tokenizer.encode(inputs)
                _sync()
                if len(image_codes.shape) == 4:
                    image_codes = image_codes.squeeze(1)
                H, W = image_codes.shape[1], image_codes.shape[2]
                indices = image_codes.cpu().numpy()
        finally:
            if self.use_llm:
                self.image_tokenizer.to("cpu")
                _sync()
                _free_cache()
        return indices, (H, W)

    def decode(self, tokens: np.ndarray, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        """VQ indices ([B, H'*W'] or flat) -> images [B, C, H, W] in [0, 1]."""
        H, W = spatial_shape
        if tokens.ndim == 1:
            B = tokens.shape[0] // (H * W)
            tokens = tokens.reshape(B, H * W)
        else:
            B = tokens.shape[0]

        device = self._vq_device()
        if self.use_llm:
            self.image_tokenizer.to(device)
            _sync()
        try:
            tokens_t = torch.from_numpy(tokens).long().to(device).reshape(B, H, W)
            with torch.no_grad():
                recon = self.image_tokenizer.decode(tokens_t)
                _sync()
                recon = torch.clamp((recon + 1.0) / 2.0, 0.0, 1.0).float()
        finally:
            if self.use_llm:
                self.image_tokenizer.to("cpu")
                _sync()
                _free_cache()
        return recon

    # ------------------------------------------------------------------
    # LMM helpers
    # ------------------------------------------------------------------

    def prepare_prefix_prompt(self, H: int, W: int, batch_size: int = 1) -> torch.Tensor:
        """Prefix ``BOS + BOI + "H*W" + IMG`` as token IDs [batch_size, L] on the device."""
        prefix_str = (
            self.tokenizer.bos_token + self.tokenizer.boi_token + f"{H}*{W}" + self.tokenizer.img_token
        )
        inputs = self.tokenizer(prefix_str, return_tensors="pt", add_special_tokens=False)
        return inputs.input_ids.to(self.device).repeat(batch_size, 1)

    def clear_llm_memory(self):
        """Move the LMM back to the CPU and release accelerator memory."""
        if self.model is not None:
            self.model.to("cpu")
            _sync()
        _free_cache()


class Emu3SoftDecoder(CommunicationMethod):
    """LMM-based joint semantic-channel token detection (greedy).

    For each token position i (raster order) the decision is

        s_i = argmax_k [ log P(y_i | s_i = k) + o_{i,k} / tau ]

    where the first term is the token log-likelihood computed from the
    information-bit LLRs and o_{i,k} is the Emu3-Gen logit for visual token k
    conditioned on the previously decided tokens. A larger temperature tau
    weakens the LMM prior.
    """

    def __init__(
        self,
        emu: Emu3Wrapper,
        tokens_per_image: int,
        spatial_shape: Tuple[int, int],
        num_bits_per_symbol: int = 2,
        name: str = "Emu3SoftDecoder",
    ):
        super().__init__(
            name=name, bits_per_token=emu.bits_per_token, num_bits_per_symbol=num_bits_per_symbol
        )
        self.emu = emu
        self.tokens_per_image = tokens_per_image
        self.spatial_shape = spatial_shape
        self.vocab_size = 2**self.bits_per_token

        self.past_key_values = None
        self.last_input_ids = None

        # VQ codebook index -> LMM token ID
        self.vq_to_llm_map = self._create_vocab_mapping()

        # Bit pattern of every token, [1, V, bits_per_token] (LSB first)
        bit_mask = np.array(
            [[(idx >> i) & 1 for i in range(self.bits_per_token)] for idx in range(self.vocab_size)],
            dtype=np.float32,
        )
        self.bit_mask_tf = tf.reshape(tf.constant(bit_mask), [1, self.vocab_size, self.bits_per_token])

    def _create_vocab_mapping(self) -> torch.Tensor:
        """Map each VQ codebook index to the corresponding LMM visual-token ID."""
        template = self.emu.processor.visual_template[0]
        ids0 = self.emu.tokenizer(template.format(token_id=0), add_special_tokens=False)["input_ids"]
        ids1 = self.emu.tokenizer(template.format(token_id=1), add_special_tokens=False)["input_ids"]

        start_id, next_id = ids0[0], ids1[0]
        if next_id == start_id + 1:
            return torch.arange(start_id, start_id + self.vocab_size, device=self.emu.device, dtype=torch.long)
        logger.warning("Visual tokens are not contiguous; building the mapping token by token.")
        strings = [template.format(token_id=i) for i in range(self.vocab_size)]
        encoded = self.emu.tokenizer(strings, add_special_tokens=False)["input_ids"]
        return torch.tensor([e[0] for e in encoded], device=self.emu.device, dtype=torch.long)

    def reset_cache(self):
        if self.past_key_values is not None:
            del self.past_key_values
        self.past_key_values = None
        self.last_input_ids = None
        _free_cache()

    # ------------------------------------------------------------------
    # Prior and likelihood
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_llm_logits(self, input_tokens, use_cache: bool = True, is_llm_ids: bool = False) -> tf.Tensor:
        """Run one LMM step and return the logits of the V visual tokens as a TF tensor [B, V].

        ``input_tokens`` are VQ indices (or LMM IDs if ``is_llm_ids``). ``None``
        feeds the stored prefix (first step).
        """
        device = self.emu.device
        if input_tokens is not None:
            if isinstance(input_tokens, (np.ndarray, tf.Tensor)):
                input_tensor = torch.from_numpy(np.array(input_tokens)).to(device).long()
            else:
                input_tensor = input_tokens.to(device).long()
            input_ids = input_tensor if is_llm_ids else self.vq_to_llm_map[input_tensor]
        else:
            input_ids = self.last_input_ids

        if self.past_key_values is None:
            seq_len = input_ids.shape[1]
        else:
            seq_len = self.past_key_values[0][0].shape[2] + input_ids.shape[1]
        attention_mask = torch.ones((input_ids.shape[0], seq_len), device=device, dtype=torch.long)

        self.emu.model.to(device)
        outputs = self.emu.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values if use_cache else None,
            use_cache=use_cache,
        )
        if use_cache:
            self.past_key_values = outputs.past_key_values

        # Raw logits of the visual tokens (no softmax)
        logits = outputs.logits[:, -1, :][:, self.vq_to_llm_map]
        return tf.convert_to_tensor(logits.cpu().float().numpy(), dtype=tf.float32)

    def _compute_step_phy_probs(self, llr_step: tf.Tensor) -> tf.Tensor:
        """Token likelihoods P(y_i | s_i = k) from bit LLRs: [B, bits] -> [B, V]."""
        prob_bit_1 = tf.nn.sigmoid(llr_step)
        prob_bit_0 = 1.0 - prob_bit_1
        p1 = tf.expand_dims(prob_bit_1, axis=1)  # [B, 1, bits]
        p0 = tf.expand_dims(prob_bit_0, axis=1)
        log_probs = tf.reduce_sum(
            tf.math.log(p1 + 1e-12) * self.bit_mask_tf
            + tf.math.log(p0 + 1e-12) * (1.0 - self.bit_mask_tf),
            axis=-1,
        )
        return tf.exp(log_probs)

    def _prepare_llr_batch(self, llr) -> Tuple[int, tf.Tensor]:
        """Reshape the LLRs to [batch, tokens_per_image, bits_per_token]."""
        llr_flat = tf.reshape(llr, [-1])
        bits_per_image = self.tokens_per_image * self.bits_per_token
        total = int(tf.size(llr_flat))
        if total < bits_per_image:
            raise ValueError(f"Not enough LLRs: {total} < {bits_per_image}")
        batch_size = total // bits_per_image
        llr_valid = llr_flat[: batch_size * bits_per_image]
        return batch_size, tf.reshape(llr_valid, [batch_size, self.tokens_per_image, self.bits_per_token])

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(self, llr, temperature: float = 1.0, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy joint detection.

        Args:
            llr: information-bit LLRs from the channel (Sionna).
            temperature: LMM logit temperature tau.
            use_cache: use the LMM KV cache.
        Returns:
            rx_bits [total_bits] float32, rx_tokens [total_tokens] int64.
        """
        self.reset_cache()
        batch_size, llr_batched = self._prepare_llr_batch(llr)
        H, W = self.spatial_shape

        self.last_input_ids = self.emu.prepare_prefix_prompt(H, W, batch_size=batch_size)
        decoded_sequence = []
        current_token_vq = None
        logger.info(
            f"Greedy joint detection: batch={batch_size}, tau={temperature}, tokens={self.tokens_per_image}"
        )

        for t in range(self.tokens_per_image):
            if t > 0 and t % W == 0:
                # New row: feed the previous token followed by the end-of-line token
                prev_torch = torch.from_numpy(current_token_vq.numpy()).to(self.emu.device).long()
                prev_llm = self.vq_to_llm_map[prev_torch]
                eol = torch.full((prev_llm.size(0), 1), self.emu.eol_token_id, device=self.emu.device)
                combined = torch.cat([prev_llm.unsqueeze(1), eol], dim=1)
                sem_logits = self._get_llm_logits(combined, use_cache=use_cache, is_llm_ids=True)
            else:
                input_for_llm = tf.expand_dims(current_token_vq, 1) if t > 0 else None
                sem_logits = self._get_llm_logits(input_for_llm, use_cache=use_cache)

            phy_probs = self._compute_step_phy_probs(llr_batched[:, t, :])

            # score(k) = log P(y_i | s_i = k) + o_{i,k} / tau
            log_phy = tf.math.log(phy_probs + 1e-12)
            combined_scores = log_phy + sem_logits / temperature

            fused_preds = tf.argmax(combined_scores, axis=-1, output_type=tf.int32).numpy()
            decoded_sequence.append(fused_preds)
            current_token_vq = tf.convert_to_tensor(fused_preds, dtype=tf.int32)

            if (t + 1) % 512 == 0:
                logger.debug(f"step {t + 1}/{self.tokens_per_image}")

        decoded_flat = tf.stack(decoded_sequence, axis=1).numpy().flatten()
        return tokens_to_bits(decoded_flat, self.bits_per_token), decoded_flat
