"""
Emu3 classes (legacy version used for the paper experiments, including beam search):
  - Emu3Wrapper       : holds the VQ tokenizer and the LMM (Emu3-Gen)
  - Emu3SoftDecoder   : soft decoder fusing the LLRs and the LMM prior
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tensorflow as tf

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Locally modified Emu3ForCausalLM with window attention.
# It used to be imported by relative path from a modified clone of baaivision/Emu3
# (2026-03-05) and was vendored into this repository for reproducibility.
# A fallback would silently ignore window_size, so a missing module is an error.
from .third_party_models.modeling_emu3 import Emu3ForCausalLM as _Emu3ForCausalLM

_USE_LOCAL_EMU3 = True
print("Local Emu3ForCausalLM imported successfully. Window attention enabled.")

from .base import CommunicationMethod, bits_to_tokens, tokens_to_bits


# ---------------------------------------------------------------------------
# Emu3Wrapper
# ---------------------------------------------------------------------------

class Emu3Wrapper:
    """
    Wrapper holding the Emu3 VQ tokenizer and the LMM.

    - encode(images_tensor) : [B,C,H,W] -> VQ indices [B,H,W]
    - decode(tokens, spatial_shape) : tokens -> images [B,C,H,W]
    - prepare_prefix_prompt(H, W, batch_size) : prefix token IDs for the LMM
    """

    def __init__(self, cfg, use_llm: bool = True):
        """
        Required cfg fields:
            emu_hub (str): Hugging Face ID of Emu3-Gen
            vq_hub  (str): Hugging Face ID of Emu3-VisionTokenizer
        use_llm:
            If False, the LMM (Emu3-Gen) is not loaded.
            The VQ tokenizer is always loaded.
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

        # VQ codebook size (32768 = 2^15 for Emu3)
        self.num_embed = 32768
        self.bits_per_token = int(np.log2(self.num_embed))

        print("Loading Emu3 VQ-VAE (CPU init)...")
        t0 = time.time()
        self.image_processor = AutoImageProcessor.from_pretrained(
            cfg.vq_hub, trust_remote_code=True
        )

        # VQ tokenizer: kept on the CPU and moved to the GPU only when used
        self.image_tokenizer = AutoModel.from_pretrained(
            cfg.vq_hub, device_map="cpu", trust_remote_code=True,
            torch_dtype=self.dtype,
        ).eval()
        t_vqvae = time.time() - t0
        print(f"[TIMING] VQ-VAE from_pretrained: {t_vqvae:.2f}s")

        if use_llm:
            print("Loading Emu3 LLM (CPU init)...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.emu_hub, trust_remote_code=True, padding_side="left"
            )

            # Emu3Processor (needed for visual_template)
            try:
                from .third_party_models.processing_emu3 import Emu3Processor
                self.processor = Emu3Processor(
                    self.image_processor, self.image_tokenizer, self.tokenizer
                )
            except ImportError:
                self.processor = None

            # LMM: the locally modified version with window attention
            t0 = time.time()
            if _USE_LOCAL_EMU3:
                print("Loading LLM with local Emu3ForCausalLM (window attention enabled)...")
                self.model = _Emu3ForCausalLM.from_pretrained(
                    cfg.emu_hub,
                    device_map="cpu",
                    torch_dtype=self.dtype,
                    trust_remote_code=False,
                    attn_implementation="eager",
                ).eval()
            else:
                print("Warning: local Emu3ForCausalLM not found. Falling back to AutoModelForCausalLM (window_size will be ignored).")
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.emu_hub,
                    device_map="cpu",
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                    attn_implementation="eager",
                ).eval()
            t_llm = time.time() - t0
            print(f"[TIMING] LLM from_pretrained: {t_llm:.2f}s")

            self.eol_token_id = self.tokenizer.eos_token_id

            # set the LMM attention window size R
            self.window_size = cfg.window_size
            self.model.model.window_size = self.window_size
            print(f"Emu3 loaded (LLM+VQ). bits_per_token={self.bits_per_token}, window_size={self.window_size}")
        else:
            # no LMM: VQ tokenizer only
            # without the LMM the tokenizer stays on the GPU
            self.tokenizer = None
            self.processor = None
            self.model = None
            self.eol_token_id = None
            self.window_size = cfg.window_size
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            t0 = time.time()
            self.image_tokenizer.to(target_device)
            print(f"[TIMING] VQ-VAE CPU→GPU transfer: {time.time() - t0:.2f}s")
            print(f"Emu3 loaded (VQ-only, LLM skipped). bits_per_token={self.bits_per_token}, image_tokenizer on {target_device}")

    # ------------------------------------------------------------------
    # VQ encode / decode
    # ------------------------------------------------------------------

    def encode(self, images_tensor: torch.Tensor) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Args:
            images_tensor: [B, C, H, W] float in [0, 1]
        Returns:
            indices: [B, H_lat, W_lat] numpy int array
            spatial_shape: (H_lat, W_lat)
        """
        pil_images = [TF.to_pil_image(img.cpu()) for img in images_tensor]

        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        # with use_llm=False the tokenizer is already on the GPU
        if self.use_llm:
            t0 = time.time()
            self.image_tokenizer.to(target_device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[TIMING] encode CPU→GPU transfer: {time.time() - t0:.3f}s")
        try:
            with torch.no_grad():
                inputs = self.image_processor(
                    pil_images, do_resize=False, return_tensors="pt"
                )["pixel_values"].to(target_device, dtype=self.image_tokenizer.dtype)

                t0 = time.time()
                image_codes = self.image_tokenizer.encode(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"[TIMING] encode inference: {time.time() - t0:.3f}s")

                if len(image_codes.shape) == 4:
                    image_codes = image_codes.squeeze(1)

                H, W = image_codes.shape[1], image_codes.shape[2]
                indices = image_codes.cpu().numpy()
        finally:
            if self.use_llm:
                t0 = time.time()
                self.image_tokenizer.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"[TIMING] encode GPU→CPU transfer: {time.time() - t0:.3f}s")
                _free_cache()

        return indices, (H, W)

    def decode(self, tokens: np.ndarray, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            tokens: [B, H*W] or [B*H*W] int numpy array
            spatial_shape: (H_lat, W_lat)
        Returns:
            images: [B, C, H_px, W_px] float in [0, 1]
        """
        H, W = spatial_shape
        if tokens.ndim == 1:
            B = tokens.shape[0] // (H * W)
            tokens = tokens.reshape(B, H * W)
        else:
            B = tokens.shape[0]

        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        # with use_llm=False the tokenizer is already on the GPU
        if self.use_llm:
            t0 = time.time()
            self.image_tokenizer.to(target_device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[TIMING] decode CPU→GPU transfer: {time.time() - t0:.3f}s")
        try:
            tokens_t = torch.from_numpy(tokens).long().to(target_device).reshape(B, H, W)
            with torch.no_grad():
                t0 = time.time()
                recon = self.image_tokenizer.decode(tokens_t)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"[TIMING] decode inference (B={B}): {time.time() - t0:.3f}s")
                recon = torch.clamp((recon + 1.0) / 2.0, 0.0, 1.0).float()
        finally:
            if self.use_llm:
                t0 = time.time()
                self.image_tokenizer.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"[TIMING] decode GPU→CPU transfer: {time.time() - t0:.3f}s")
                _free_cache()

        return recon

    # ------------------------------------------------------------------
    # LLM prefix prompt
    # ------------------------------------------------------------------

    def prepare_prefix_prompt(self, H: int, W: int, batch_size: int = 1) -> torch.Tensor:
        """Build BOS + BOI + HxW + IMG token IDs on the GPU."""
        prefix_str = (
            self.tokenizer.bos_token
            + self.tokenizer.boi_token
            + f"{H}*{W}"
            + self.tokenizer.img_token
        )
        inputs = self.tokenizer(prefix_str, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.device)
        return input_ids.repeat(batch_size, 1)

    def clear_llm_memory(self):
        """Move the LMM to the CPU and free GPU memory (no-op without the LMM)."""
        if self.model is not None:
            self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # wait for GPU kernels before freeing
        _free_cache()


# ---------------------------------------------------------------------------
# Emu3SoftDecoder
# ---------------------------------------------------------------------------

class Emu3SoftDecoder(CommunicationMethod):
    """
    Detect tokens by fusing the physical-layer LLRs with the Emu3 LMM logits.

    scoring: log P(y_i | s_i=k)  +  o_{i,k} / τ
        - first term: token log-likelihood from the LLRs
        - second term: raw LMM logits scaled by 1/tau (larger tau = weaker prior)

    Call decode(llr, temperature):
        llr        : [total_bits] or [batch, tokens, bits] float (Sionna)
        temperature: temperature tau of the LMM logits (default 1.0)
    Returns:
        rx_bits  : [total_bits] numpy float32
        rx_tokens: [total_tokens] numpy int64
    """

    def __init__(
        self,
        emu: Emu3Wrapper,
        tokens_per_image: int,
        spatial_shape: Tuple[int, int],
        num_bits_per_symbol: int = 2,
        beam_width: int = 1,
        name: str = "Emu3SoftDecoder",
    ):
        super().__init__(
            name=name,
            bits_per_token=emu.bits_per_token,
            num_bits_per_symbol=num_bits_per_symbol,
        )
        self.emu = emu
        self.tokens_per_image = tokens_per_image
        self.spatial_shape = spatial_shape
        self.beam_width = beam_width

        self.past_key_values = None
        self.last_input_ids = None

        # VQ index -> LMM token ID mapping
        self.vq_to_llm_map = self._create_vocab_mapping()

        # precomputed bit masks [1, Vocab, Bits]
        vocab_size = 2 ** self.bits_per_token
        self.vocab_size = vocab_size
        indices = np.arange(vocab_size)
        bit_mask = np.array(
            [[(idx >> i) & 1 for i in range(self.bits_per_token)] for idx in range(vocab_size)],
            dtype=np.float32,
        )
        self.bit_mask_tf = tf.constant(bit_mask)
        self.bit_mask_tf = tf.reshape(self.bit_mask_tf, [1, vocab_size, self.bits_per_token])

    # ------------------------------------------------------------------
    # Vocab mapping
    # ------------------------------------------------------------------

    def _create_vocab_mapping(self) -> torch.Tensor:
        """Map VQ codebook indices to LMM token IDs."""
        print("Creating VQ → LLM token mapping...")
        vocab_size = 2 ** self.bits_per_token

        if self.emu.processor is None:
            raise RuntimeError(
                "Emu3Wrapper.processor is None; initialize Emu3Processor."
            )

        visual_template = self.emu.processor.visual_template[0]
        ids0 = self.emu.tokenizer(
            visual_template.format(token_id=0), add_special_tokens=False
        )["input_ids"]
        ids1 = self.emu.tokenizer(
            visual_template.format(token_id=1), add_special_tokens=False
        )["input_ids"]

        start_id, next_id = ids0[0], ids1[0]
        if next_id == start_id + 1:
            mapping = torch.arange(
                start_id, start_id + vocab_size, device=self.emu.device, dtype=torch.long
            )
        else:
            print("Warning: Visual tokens are not sequential. Using slow mapping.")
            batch_strings = [visual_template.format(token_id=i) for i in range(vocab_size)]
            encoded = self.emu.tokenizer(batch_strings, add_special_tokens=False)["input_ids"]
            mapping = torch.tensor(
                [e[0] for e in encoded], device=self.emu.device, dtype=torch.long
            )

        return mapping

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reset_cache(self):
        if self.past_key_values is not None:
            del self.past_key_values
        self.past_key_values = None
        self.last_input_ids = None
        _free_cache()

    # ------------------------------------------------------------------
    # LMM logits
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_llm_logits(
        self,
        input_tokens,
        use_cache: bool = True,
        is_llm_ids: bool = False,
        save_attention: bool = False,
        step_idx: int = -1,
        vis_logdir: Optional[str] = None,
    ) -> Tuple[tf.Tensor, object]:
        device = self.emu.device

        if input_tokens is not None:
            if isinstance(input_tokens, (np.ndarray, tf.Tensor)):
                input_tensor = torch.from_numpy(np.array(input_tokens)).to(device).long()
            else:
                input_tensor = input_tokens.to(device).long()

            input_ids = input_tensor if is_llm_ids else self.vq_to_llm_map[input_tensor]
        else:
            input_ids = self.last_input_ids

        # Attention mask
        if self.past_key_values is None:
            seq_len = input_ids.shape[1]
        else:
            past_seq_len = self.past_key_values[0][0].shape[2]
            seq_len = past_seq_len + input_ids.shape[1]
        attention_mask = torch.ones(
            (input_ids.shape[0], seq_len), device=device, dtype=torch.long
        )

        self.emu.model.to(device)
        try:
            outputs = self.emu.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=self.past_key_values if use_cache else None,
                use_cache=use_cache,
                output_attentions=save_attention,
            )
        except Exception as e:
            print(f"Error during model forward: {e}")
            raise

        # optionally save attention maps
        if save_attention and vis_logdir and outputs.attentions is not None:
            q_len = input_ids.shape[1]
            target_dir = os.path.join(vis_logdir, "attention_raw")
            os.makedirs(target_dir, exist_ok=True)
            last_layer_attn = outputs.attentions[-1][:, :, -q_len:, :].detach().cpu()
            fname = f"step_{step_idx:04d}.pt" if step_idx >= 0 else "prefill.pt"
            torch.save(last_layer_attn, os.path.join(target_dir, fname))

        if use_cache:
            self.past_key_values = outputs.past_key_values

        # logits of the visual tokens as a TF tensor (no softmax)
        next_token_logits = outputs.logits[:, -1, :]
        relevant_logits = next_token_logits[:, self.vq_to_llm_map]
        logits_tf = tf.convert_to_tensor(relevant_logits.cpu().float().numpy(), dtype=tf.float32)

        return logits_tf, self.past_key_values

    # ------------------------------------------------------------------
    # physical-layer likelihoods
    # ------------------------------------------------------------------

    def _compute_step_phy_probs(self, llr_step: tf.Tensor) -> tf.Tensor:
        """LLR [Batch, Bits] -> token likelihoods [Batch, Vocab]."""
        prob_bit_1 = tf.nn.sigmoid(llr_step)
        prob_bit_0 = 1.0 - prob_bit_1
        p1 = tf.expand_dims(prob_bit_1, axis=1)  # [B, 1, Bits]
        p0 = tf.expand_dims(prob_bit_0, axis=1)
        log_probs = tf.reduce_sum(
            tf.math.log(p1 + 1e-12) * self.bit_mask_tf
            + tf.math.log(p0 + 1e-12) * (1.0 - self.bit_mask_tf),
            axis=-1,
        )
        return tf.exp(log_probs)

    # ------------------------------------------------------------------
    # KV cache utilities
    # ------------------------------------------------------------------

    def _inject_eol_to_cache(self, batch_size: int):
        """Feed the EOL token to the LMM to update the KV cache."""
        eol_input = (
            torch.tensor([[self.emu.eol_token_id]], device=self.emu.device)
            .repeat(batch_size, 1)
        )
        with torch.no_grad():
            self._get_llm_logits(eol_input, use_cache=True, is_llm_ids=True)

    def _reorder_cache(self, past_key_values, beam_indices: torch.Tensor):
        if past_key_values is None:
            return None
        return tuple(
            tuple(state.index_select(0, beam_indices) for state in layer_past)
            for layer_past in past_key_values
        )

    # ------------------------------------------------------------------
    # LLR batching
    # ------------------------------------------------------------------

    def _prepare_llr_batch(
        self, llr
    ) -> Tuple[int, tf.Tensor]:
        """Reshape the LLRs to [Batch, Tokens, Bits]."""
        llr_flat = tf.reshape(llr, [-1])
        total_bits_needed = self.tokens_per_image * self.bits_per_token
        total_available = int(tf.size(llr_flat))

        if total_available < total_bits_needed:
            raise ValueError(
                f"Not enough LLRs: {total_available} < {total_bits_needed}"
            )

        batch_size = total_available // total_bits_needed
        llr_valid = llr_flat[: batch_size * total_bits_needed]
        return batch_size, tf.reshape(
            llr_valid, [batch_size, self.tokens_per_image, self.bits_per_token]
        )

    # ------------------------------------------------------------------
    # bit conversion
    # ------------------------------------------------------------------

    def _tokens_to_bits(self, tokens: np.ndarray) -> np.ndarray:
        return tokens_to_bits(tokens, self.bits_per_token)

    # ------------------------------------------------------------------
    # greedy decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_greedy(
        self,
        llr,
        temperature: float,
        use_cache: bool,
        vis_logdir: Optional[str],
        vis_sample_index: int,
        save_attention: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.reset_cache()
        batch_size, llr_batched = self._prepare_llr_batch(llr)
        H, W = self.spatial_shape

        prefix_ids = self.emu.prepare_prefix_prompt(H, W, batch_size=batch_size)
        self.last_input_ids = prefix_ids

        decoded_sequence = []
        current_token_vq = None

        print(f"[Greedy] batch={batch_size}, temperature={temperature}, tokens={self.tokens_per_image}")

        for t in range(self.tokens_per_image):
            # new row: feed the previous token and EOL
            if t > 0 and t % W == 0:
                prev_torch = torch.from_numpy(current_token_vq.numpy()).to(self.emu.device).long()
                prev_llm = self.vq_to_llm_map[prev_torch]
                eol = torch.full(
                    (prev_llm.size(0), 1), self.emu.eol_token_id, device=self.emu.device
                )
                combined = torch.cat([prev_llm.unsqueeze(1), eol], dim=1)
                sem_logits, _ = self._get_llm_logits(
                    combined, use_cache=use_cache, is_llm_ids=True,
                    save_attention=save_attention, step_idx=t, vis_logdir=vis_logdir,
                )
            else:
                input_for_llm = tf.expand_dims(current_token_vq, 1) if t > 0 else None
                sem_logits, _ = self._get_llm_logits(
                    input_for_llm, use_cache=use_cache,
                    save_attention=save_attention, step_idx=t, vis_logdir=vis_logdir,
                )

            phy_probs = self._compute_step_phy_probs(llr_batched[:, t, :])

            # score(k) = log P(y_i | s_i=k)  +  o_{i,k} / τ
            log_phy = tf.math.log(phy_probs + 1e-12)
            combined_scores = log_phy + sem_logits / temperature

            fused_preds = tf.argmax(combined_scores, axis=-1, output_type=tf.int32).numpy()
            decoded_sequence.append(fused_preds)
            current_token_vq = tf.convert_to_tensor(fused_preds, dtype=tf.int32)

            if (t + 1) % 100 == 0:
                print(f"  [step {t+1:04d}/{self.tokens_per_image}]")

        decoded_batched = tf.stack(decoded_sequence, axis=1).numpy()
        decoded_flat = decoded_batched.flatten()
        return self._tokens_to_bits(decoded_flat), decoded_flat

    # ------------------------------------------------------------------
    # beam-search decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_beam_search(
        self,
        llr,
        temperature: float,
        vis_logdir: Optional[str],
        vis_sample_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size, llr_batched = self._prepare_llr_batch(llr)
        H, W = self.spatial_shape
        K = self.beam_width
        device = self.emu.device
        self.reset_cache()

        print(f"[Beam] batch={batch_size}, K={K}, temperature={temperature}")

        beam_scores = torch.full((batch_size, K), -float("inf"), device=device)
        beam_scores[:, 0] = 0.0
        generated = torch.zeros(
            (batch_size, K, self.tokens_per_image), dtype=torch.long, device=device
        )

        prefix_ids = self.emu.prepare_prefix_prompt(H, W, batch_size=batch_size)
        self.last_input_ids = prefix_ids.unsqueeze(1).expand(-1, K, -1).reshape(
            batch_size * K, -1
        )
        with torch.no_grad():
            self._get_llm_logits(None, use_cache=True)

        current_token_vq = None

        for t in range(self.tokens_per_image):
            if t > 0 and t % W == 0:
                self._inject_eol_to_cache(batch_size * K)

            input_for_llm = tf.expand_dims(current_token_vq, 1) if t > 0 else None
            sem_logits_tf, _ = self._get_llm_logits(input_for_llm, use_cache=True)

            llr_step = llr_batched[:, t, :]
            llr_step_expanded = tf.repeat(llr_step, repeats=K, axis=0)
            phy_probs_tf = self._compute_step_phy_probs(llr_step_expanded)

            sem_logits = torch.from_numpy(sem_logits_tf.numpy()).to(device)
            phy_probs = torch.from_numpy(phy_probs_tf.numpy()).to(device)

            # score(k) = log P(y_i | s_i=k)  +  o_{i,k} / τ
            step_scores = torch.log(phy_probs + 1e-12) + sem_logits / temperature
            total_scores = beam_scores.view(batch_size * K, 1) + step_scores

            total_scores_r = total_scores.view(batch_size, K * self.vocab_size)
            topk_scores, topk_indices = torch.topk(total_scores_r, k=K, dim=1)

            beam_from = topk_indices.div(self.vocab_size, rounding_mode="floor")
            token_idx = topk_indices % self.vocab_size

            batch_inds = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, K)
            generated = generated[batch_inds, beam_from, :]
            generated[:, :, t] = token_idx
            beam_scores = topk_scores

            # reorder the KV cache
            base = torch.arange(0, batch_size * K, step=K, device=device).unsqueeze(1)
            reorder_idx = (base + beam_from).view(-1)
            std_idx = torch.arange(batch_size * K, device=device)
            if not torch.equal(reorder_idx, std_idx):
                self.past_key_values = self._reorder_cache(self.past_key_values, reorder_idx)

            current_token_vq = tf.convert_to_tensor(token_idx.view(-1).tolist(), dtype=tf.int32)

            if (t + 1) % 100 == 0:
                print(f"  [step {t+1:04d}/{self.tokens_per_image}]")

        best = generated[:, 0, :].cpu().numpy()
        decoded_flat = best.flatten()
        return self._tokens_to_bits(decoded_flat), decoded_flat

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def decode(
        self,
        llr,
        temperature: float = 1.0,
        use_cache: bool = True,
        vis_logdir: Optional[str] = None,
        vis_sample_index: int = 0,
        save_attention: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            llr        : LLRs from the Sionna demapper
            temperature: temperature tau of the LMM logits.
                         Larger values weaken the prior,
                         smaller values strengthen it.
        Returns:
            rx_bits  : [total_bits] numpy float32
            rx_tokens: [total_tokens] numpy int64
        """
        if self.beam_width > 1:
            return self._decode_beam_search(llr, temperature, vis_logdir, vis_sample_index)
        else:
            return self._decode_greedy(
                llr, temperature, use_cache, vis_logdir, vis_sample_index, save_attention
            )


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def _free_cache():
    import gc
    gc.collect()
    gc.collect()  # reference cycles are not always collected in one pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
