"""Bit/token conversion and the Sionna-based modulation / channel-coding blocks."""

from typing import Tuple

import numpy as np
import sionna as sn
import tensorflow as tf


def bits_to_tokens(bits: np.ndarray, bits_per_token: int) -> np.ndarray:
    """Convert a flat bit array (LSB first) into token indices."""
    bits_reshaped = bits.reshape(-1, bits_per_token).astype(np.int64)
    powers_of_two = 2 ** np.arange(bits_per_token, dtype=np.int64)
    return np.dot(bits_reshaped, powers_of_two)


def tokens_to_bits(tokens: np.ndarray, bits_per_token: int) -> np.ndarray:
    """Convert token indices into a flat bit array (LSB first)."""
    tokens = np.array(tokens, dtype=np.int64)[:, np.newaxis]
    bit_shifts = np.arange(bits_per_token)
    bits = (tokens >> bit_shifts) & 1
    return bits.ravel().astype(np.float32)


class CommunicationMethod:
    """Uncoded transmission with QAM mapping and APP demapping (Sionna)."""

    def __init__(
        self,
        name: str,
        bits_per_token: int,
        num_bits_per_symbol: int,
        coderate: float = 1.0,
    ):
        self.name = name
        self.bits_per_token = bits_per_token
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coderate = coderate

        self.constellation = sn.mapping.Constellation("qam", num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper(
            demapping_method="app", constellation=self.constellation
        )

    def encode(self, tx_bits_flat) -> tf.Tensor:
        """No channel coding: reshape the bits to [1, N]."""
        if not isinstance(tx_bits_flat, tf.Tensor):
            tx_bits_flat = tf.convert_to_tensor(tx_bits_flat, dtype=tf.float32)
        return tf.reshape(tx_bits_flat, [1, -1])

    def decode(self, llr) -> Tuple[np.ndarray, np.ndarray]:
        """Hard decision on the LLRs. Returns (rx_bits, rx_tokens)."""
        rx_bits = tf.cast(llr > 0, tf.float32).numpy().flatten()
        rx_tokens = bits_to_tokens(rx_bits, self.bits_per_token)
        return rx_bits, rx_tokens


class LDPCCommunicationMethod(CommunicationMethod):
    """5G NR LDPC coding with belief-propagation (BP) decoding.

    The codeword length ``n`` is given and the number of information bits is
    derived as ``k = int(coderate * n)``.

    - ``encode`` splits the information bits into blocks of ``k`` bits
      (zero-padding the last block) and returns the codewords as [1, N_coded].
    - ``ldpc_decode`` runs BP on the channel LLRs and returns the soft
      a-posteriori LLRs of the ``N_info`` information bits. It is called by
      the channel right after demapping, so the receiver only sees
      information-bit LLRs.
    """

    def __init__(
        self,
        name: str,
        bits_per_token: int,
        num_bits_per_symbol: int,
        n: int = 2048,
        coderate: float = 0.5,
        num_iter: int = 20,
    ):
        k = int(coderate * n)
        super().__init__(name, bits_per_token, num_bits_per_symbol, coderate=k / n)
        self.k = k
        self.n = n
        self.ldpc_encoder = sn.fec.ldpc.LDPC5GEncoder(k=k, n=n)
        # hard_out=False: return soft LLRs (needed for the soft-decision fusion)
        # return_infobits=True: return only the k information-bit LLRs
        self.ldpc_decoder = sn.fec.ldpc.LDPC5GDecoder(
            self.ldpc_encoder, num_iter=num_iter, hard_out=False, return_infobits=True
        )

    def encode(self, tx_bits_flat) -> tf.Tensor:
        """LDPC-encode the information bits and return [1, N_coded]."""
        if not isinstance(tx_bits_flat, tf.Tensor):
            tx_bits_flat = tf.convert_to_tensor(tx_bits_flat, dtype=tf.float32)
        tx_bits_flat = tf.reshape(tx_bits_flat, [-1])
        n_info = int(tf.size(tx_bits_flat))

        # Zero-pad to a multiple of k
        num_codewords = (n_info + self.k - 1) // self.k
        n_padded = num_codewords * self.k
        if n_padded > n_info:
            padding = tf.zeros([n_padded - n_info], dtype=tf.float32)
            tx_bits_flat = tf.concat([tx_bits_flat, padding], axis=0)

        info_blocks = tf.reshape(tx_bits_flat, [num_codewords, self.k])
        coded_blocks = self.ldpc_encoder(info_blocks)  # [num_codewords, n]
        return tf.reshape(coded_blocks, [1, -1])

    def ldpc_decode(self, llr: tf.Tensor, n_info: int) -> tf.Tensor:
        """BP-decode channel LLRs and return the information-bit LLRs [n_info]."""
        llr_flat = tf.reshape(llr, [-1])
        num_codewords = int(tf.size(llr_flat)) // self.n
        llr_blocks = tf.reshape(llr_flat[: num_codewords * self.n], [num_codewords, self.n])
        info_llr_blocks = self.ldpc_decoder(llr_blocks)  # [num_codewords, k]
        return tf.reshape(info_llr_blocks, [-1])[:n_info]
