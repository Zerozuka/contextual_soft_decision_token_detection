#!/usr/bin/env python3
"""sweep_window_fast.py
SNR sweep for several window sizes.
The models are loaded once and, for each window size,
  emu.model.model.window_size is changed,
so the start-up cost is paid only once.

Usage (from the repository root):
    python scripts/sweep_window_fast.py
    python scripts/sweep_window_fast.py --snr_list "-2 0 2 4 6 8 10" --window_size_list "10 128 512 4096"
    python scripts/sweep_window_fast.py --num_images 10 --tag myexp

Options:
    --snr_list          space-separated SNR values (default: "-2 0 2 4 6 8 10")
    --window_size_list  space-separated window sizes (default: "10 128 512 4096")
    --temperature       LMM temperature tau (default: 1.0)
    --num_images        number of images (default: 5)
    --batch_size        batch size (default: 5)
    --tag               suffix of the output directory
    --out_dir           output directory (default: outputs/sweep_window_<timestamp>)
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
import wandb
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), "src")
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
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
    window_size: int,
    channel: Channel,
    transmitter: Transmitter,
    receiver: Receiver,
    dataloader,
    cfg,
    out_dir: str,
    num_images: int,
    temperature: float,
) -> dict:
    """Evaluate one (SNR, window_size) point and return avg_metrics."""
    channel.cfg.snr = snr

    all_metrics = []
    all_orig, all_recon = [], []
    all_token_correct = []
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
        all_token_correct.append(tx_tokens[:L_tok] == rx_tokens[:L_tok])

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
        logger.info(f"[SNR={snr:+.1f}, R={window_size}] Batch {batch_idx:03d}: {log_str}")

        images_processed += B

    avg = {"snr": snr}
    avg.update({k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]})
    avg["fid"]         = calculate_fid(torch.stack(all_recon), torch.stack(all_orig))
    avg["temperature"] = temperature
    avg["window_size"] = window_size
    avg["use_llm"]     = True
    avg["use_ldpc"]    = cfg.channel.get("use_ldpc", False)
    avg["beam_width"]  = cfg.receiver.get("beam_width", 1)
    avg["dataset"]     = cfg.dataset.name
    avg["resolution"]  = cfg.dataset.resolution

    os.makedirs(out_dir, exist_ok=True)
    save_metrics_to_csv([avg], os.path.join(out_dir, "avg_metrics.csv"))
    np.save(os.path.join(out_dir, "token_correct.npy"),
            np.concatenate(all_token_correct))
    logger.info(
        f"[SNR={snr:+.1f}, R={window_size}] "
        f"BER={avg['ber']:.4g}  TER={avg['ter']:.4g}  -> {out_dir}"
    )
    return avg


# ---------------------------------------------------------------------------

def set_window_size(emu: Emu3Wrapper, window_size: int):
    """Change the LMM window size in place (no reload)."""
    emu.window_size = window_size
    if emu.model is not None and hasattr(emu.model, "model"):
        emu.model.model.window_size = window_size
    logger.info(f"window_size → {window_size}")


def reset_receiver_decoder(receiver: Receiver):
    """Clear the cached decoder of the Receiver so that it is rebuilt."""
    receiver._decoder = None
    receiver._decoder_key = None


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_list",          type=str, default="1 3 5 7 9")
    parser.add_argument("--window_size_list",   type=str, default="10 128 512 4096")
    parser.add_argument("--temperature",        type=float, default=1.0)
    parser.add_argument("--num_images",         type=int,   default=100)
    parser.add_argument("--batch_size",         type=int,   default=5)
    parser.add_argument("--tag",                type=str,   default="window_sweep")
    parser.add_argument("--out_dir",            type=str,   default=None)
    args = parser.parse_args()

    snr_list         = [float(x) for x in args.snr_list.split()]
    window_size_list = [int(x)   for x in args.window_size_list.split()]

    setup_logger("INFO")
    wandb.init(mode="disabled")

    # load config
    cfg = OmegaConf.load(os.path.join(ROOT_DIR, "config.yaml"))
    cfg.receiver.use_llm  = True
    cfg.receiver.temperature = args.temperature
    cfg.dataset.batch_size   = args.batch_size
    L.seed_everything(cfg.seed)

    import sionna
    sionna.config.seed = cfg.seed
    torch.set_float32_matmul_precision("high")

    # output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    sweep_dir = args.out_dir or os.path.join(
        ROOT_DIR, f"outputs/sweep_window_{timestamp}{tag_suffix}"
    )
    os.makedirs(sweep_dir, exist_ok=True)
    logger.info(f"Sweep dir      : {sweep_dir}")
    logger.info(f"SNR list       : {snr_list}")
    logger.info(f"window_size R  : {window_size_list}")
    logger.info(f"temperature τ  : {args.temperature}")
    logger.info(f"num_images     : {args.num_images}")

    # save the sweep settings
    with open(os.path.join(sweep_dir, "sweep_config.txt"), "w") as f:
        f.write(f"timestamp:       {datetime.now()}\n")
        f.write(f"snr_list:        {snr_list}\n")
        f.write(f"window_size_list:{window_size_list}\n")
        f.write(f"temperature:     {args.temperature}\n")
        f.write(f"num_images:      {args.num_images}\n")
        f.write(f"batch_size:      {args.batch_size}\n")

    # dataset (once)
    preprocess = lambda img: resize_and_crop_to_tensor(img, cfg.dataset.resolution)
    dataset    = get_test_dataset(cfg, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
    logger.info(f"Dataset: {len(dataset)} images, batch_size={cfg.dataset.batch_size}")

    # models (once)
    logger.info("Loading models (LLM + VQ-VAE)...")
    emu         = Emu3Wrapper(cfg.emu3, use_llm=True)
    transmitter = Transmitter(cfg.transmitter, emu)
    channel     = Channel(cfg.channel, emu)
    receiver    = Receiver(cfg.receiver, emu)
    logger.info("Models loaded.")

    # ---- main loop: window_size x SNR ----
    for win in window_size_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"  window_size R = {win}")
        logger.info(f"{'='*60}")

        set_window_size(emu, win)
        reset_receiver_decoder(receiver)  # rebuild the decoder for the new window size

        for snr in snr_list:
            run_name = f"snr{snr}_llmtrue_tau{args.temperature}_win{win}_ldpcfalse"
            out_dir  = os.path.join(sweep_dir, run_name)
            logger.info(f"--- [snr={snr:+.1f}, R={win}] → {run_name} ---")

            run_one_snr(
                snr=snr,
                window_size=win,
                channel=channel,
                transmitter=transmitter,
                receiver=receiver,
                dataloader=dataloader,
                cfg=cfg,
                out_dir=out_dir,
                num_images=args.num_images,
                temperature=args.temperature,
            )

    logger.info(f"\nDone. Results: {sweep_dir}")
    logger.info(
        f"Analyze: python scripts/analyze_sweep.py --results_dir {sweep_dir}"
    )


if __name__ == "__main__":
    main()
