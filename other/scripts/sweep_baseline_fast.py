#!/usr/bin/env python3
"""sweep_baseline_fast.py
SNR sweep of the hard-decision baseline (use_llm=False) with and without LDPC.
The models are loaded once and channel.cfg.snr is changed inside the SNR loop,
so the start-up cost (TensorFlow, VQ tokenizer) is paid only once.

Usage (from the repository root):
    python scripts/sweep_baseline_fast.py
    python scripts/sweep_baseline_fast.py --snr_min -2 --snr_max 8 --snr_step 0.5
    python scripts/sweep_baseline_fast.py --no_ldpc_only   # without LDPC only
    python scripts/sweep_baseline_fast.py --ldpc_only      # with LDPC only
"""

import argparse
import os
import sys
import time
from datetime import datetime

# configure TensorFlow GPU memory before importing Sionna
_t_tf_start = time.time()
import tensorflow as tf
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)
print(f"[TIMING] TensorFlow import + GPU init: {time.time() - _t_tf_start:.2f}s")

import lightning as L
import numpy as np
import torch
import torchvision.transforms.functional as TF
import wandb
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "src")
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SRC_DIR)

from data.dataset import get_test_dataset
from models import Receiver, Transmitter
from models.channel import Channel
from models.emu3 import Emu3Wrapper
from utils.csv_utils import save_metrics_to_csv
from utils.image_utils import resize_and_crop_to_tensor
from utils.logger import setup_logger
from utils.metrics import calculate_all_metrics, calculate_fid


# ---------------------------------------------------------------------------

def run_one_snr(
    snr: float,
    channel: Channel,
    transmitter: Transmitter,
    receiver: Receiver,
    dataloader,
    cfg,
    out_dir: str,
    num_images: int,
) -> dict:
    """Evaluate one SNR point and return avg_metrics."""
    # change only the SNR (the Channel is not rebuilt)
    channel.cfg.snr = snr

    all_metrics = []
    all_orig, all_recon = [], []
    images_processed = 0

    for batch_idx, data in enumerate(dataloader):
        if num_images > 0 and images_processed >= num_images:
            break

        original_batch = data["image"]
        B = original_batch.shape[0]

        start = time.time()
        transmitted, tx_times = transmitter(original_batch)
        tx_elapsed = time.time() - start

        orig_size = channel.calculate_size_KB(original_batch)
        transmitted, trans_sizes = channel(transmitted)

        start = time.time()
        recon_tensor, rx_times, rx_data = receiver(transmitted)
        rx_elapsed = time.time() - start

        for b in range(B):
            all_orig.append(original_batch[b].cpu())
            all_recon.append(recon_tensor[b].cpu())

        tx_tokens = transmitted["tx_tokens"]
        tx_bits   = transmitted["tx_bits"]
        rx_tokens = rx_data["rx_tokens"]
        rx_bits   = rx_data["rx_bits"]
        L_tok = min(len(tx_tokens), len(rx_tokens))
        L_bit = min(len(tx_bits),   len(rx_bits))
        ter = float(np.mean(tx_tokens[:L_tok] != rx_tokens[:L_tok]))
        ber = float(np.mean(tx_bits[:L_bit]   != rx_bits[:L_bit]))

        metrics = calculate_all_metrics(original_batch, recon_tensor)
        metrics.update({
            "ter": ter,
            "ber": ber,
            "tx_time": tx_elapsed,
            "rx_time": rx_elapsed,
            "compression_ratio": trans_sizes["total"] / orig_size,
        })
        metrics.update({f"tx_{k}_time": v for k, v in tx_times.items()})
        metrics.update({f"rx_{k}_time": v for k, v in rx_times.items()})
        all_metrics.append(metrics)

        log_str = {k: f"{v:.4g}" for k, v in metrics.items()}
        logger.info(f"[SNR={snr:+.1f}] Batch {batch_idx:03d}: {log_str}")

        images_processed += B

    avg = {"snr": snr}
    avg.update({k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]})
    avg["fid"] = calculate_fid(torch.stack(all_recon), torch.stack(all_orig))
    avg["temperature"] = cfg.receiver.get("temperature", 1.0)
    avg["window_size"] = cfg.emu3.window_size if cfg.receiver.get("use_llm", False) else 0
    avg["use_llm"] = False
    avg["beam_width"] = cfg.receiver.get("beam_width", 1)
    avg["dataset"] = cfg.dataset.name
    avg["resolution"] = cfg.dataset.resolution

    os.makedirs(out_dir, exist_ok=True)
    save_metrics_to_csv([avg], os.path.join(out_dir, "avg_metrics.csv"))
    logger.info(f"[SNR={snr:+.1f}] BER={avg['ber']:.4g}  TER={avg['ter']:.4g}  -> {out_dir}")
    return avg


# ---------------------------------------------------------------------------

def build_channel(cfg, emu: Emu3Wrapper, use_ldpc: bool) -> Channel:
    ch_cfg = OmegaConf.create(OmegaConf.to_container(cfg.channel, resolve=True))
    ch_cfg.use_ldpc = use_ldpc
    return Channel(ch_cfg, emu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_min",   type=float, default=-2.0)
    parser.add_argument("--snr_max",   type=float, default=8.0)
    parser.add_argument("--snr_step",  type=float, default=0.5)
    parser.add_argument("--num_images", type=int,  default=30)
    parser.add_argument("--no_ldpc_only", action="store_true")
    parser.add_argument("--ldpc_only",    action="store_true")
    parser.add_argument("--out_dir",  type=str, default=None)
    args = parser.parse_args()

    setup_logger("INFO")
    wandb.init(mode="disabled")

    # load config
    cfg = OmegaConf.load(os.path.join(ROOT_DIR, "config.yaml"))
    cfg.receiver.use_llm = False
    L.seed_everything(cfg.seed)

    import sionna
    sionna.config.seed = cfg.seed
    torch.set_float32_matmul_precision("high")

    # output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.out_dir or os.path.join(ROOT_DIR, f"outputs/sweep_baseline_fast_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)
    logger.info(f"Sweep dir: {sweep_dir}")

    # SNR list
    n_steps = round((args.snr_max - args.snr_min) / args.snr_step)
    snr_list = [round(args.snr_min + i * args.snr_step, 6) for i in range(n_steps + 1)]
    logger.info(f"SNR list: {snr_list}")

    # dataset (once)
    preprocess = lambda img: resize_and_crop_to_tensor(img, cfg.dataset.resolution)
    dataset = get_test_dataset(cfg, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
    logger.info(f"Dataset: {len(dataset)} images, batch_size={cfg.dataset.batch_size}")

    # models (once)
    logger.info("Loading models...")
    emu = Emu3Wrapper(cfg.emu3, use_llm=False)
    transmitter = Transmitter(cfg.transmitter, emu)
    receiver = Receiver(cfg.receiver, emu)
    logger.info("Models loaded.")

    run_no_ldpc = not args.ldpc_only
    run_ldpc    = not args.no_ldpc_only

    # === Phase 1: No LDPC ===
    if run_no_ldpc:
        logger.info("=== Phase 1: No LDPC ===")
        channel = build_channel(cfg, emu, use_ldpc=False)
        for snr in snr_list:
            out_dir = os.path.join(sweep_dir, f"baseline_snr{snr}")
            run_one_snr(snr, channel, transmitter, receiver, dataloader, cfg, out_dir, args.num_images)

    # === Phase 2: With LDPC ===
    if run_ldpc:
        logger.info("=== Phase 2: With LDPC ===")
        channel = build_channel(cfg, emu, use_ldpc=True)
        for snr in snr_list:
            out_dir = os.path.join(sweep_dir, f"ldpc_snr{snr}")
            run_one_snr(snr, channel, transmitter, receiver, dataloader, cfg, out_dir, args.num_images)

    logger.info(f"Done. Results: {sweep_dir}")
    logger.info("Plot: python scripts/plot_baseline_fine.py --sweep_dir " + sweep_dir)


if __name__ == "__main__":
    main()
