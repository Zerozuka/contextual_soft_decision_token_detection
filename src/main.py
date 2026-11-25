"""Run one simulation: FFHQ images -> Emu3 tokens -> AWGN channel -> token detection -> metrics.

Usage (from the repository root):
    python src/main.py channel.snr=3 receiver.use_llm=true receiver.temperature=1.0

Outputs (in the Hydra run directory ``outputs/<date>/<time>/``):
    avg_metrics.csv      averaged metrics of the run (one row)
    token_correct.npy    per-token correctness flags (bool), in transmission order
    results/recon_*.png  reconstructed images (if ``save_images=true``)
    main.log             log
"""

import os
import time

# Sionna runs on TensorFlow, which by default grabs all GPU memory.
# Enable memory growth right after the import so that PyTorch can share the GPU.
import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

import hydra
import lightning as L
import numpy as np
import pandas as pd
import sionna
import torch
import torchvision.transforms.functional as TF
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import get_test_dataset
from models import Receiver, Transmitter
from models.channel import Channel
from models.emu3 import Emu3Wrapper
from utils.image_utils import resize_and_crop_to_tensor
from utils.logger import setup_logger
from utils.metrics import calculate_all_metrics, calculate_fid


@hydra.main(config_path="../", version_base="1.1", config_name="config")
@logger.catch
def main(cfg: DictConfig):
    # 1. Setup
    setup_logger("INFO")
    print(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed)
    sionna.config.seed = cfg.seed
    torch.set_float32_matmul_precision("high")

    # 2. Data
    preprocess = lambda img: resize_and_crop_to_tensor(img, cfg.dataset.resolution)  # noqa: E731
    dataset = get_test_dataset(cfg, transform=preprocess)
    logger.info(f"Dataset {cfg.dataset.name}: {len(dataset)} images")
    dataloader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    # 3. Optional Weights & Biases logging
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True))
    else:
        wandb.init(mode="disabled")

    # 4. Models. Without the LMM (hard-decision baseline) only the VQ tokenizer is loaded.
    use_llm = cfg.receiver.get("use_llm", True)
    emu = Emu3Wrapper(cfg.emu3, use_llm=use_llm)
    transmitter = Transmitter(cfg.transmitter, emu)
    channel = Channel(cfg.channel, emu)
    receiver = Receiver(cfg.receiver, emu)

    # 5. Main loop
    all_metrics = []
    all_orig_images = []  # for FID
    all_recon_images = []  # for FID
    all_token_correct = []  # per-token correctness flags
    images_processed = 0

    pbar = tqdm(dataloader, desc="Processing batches")
    for batch_idx, data in enumerate(pbar):
        if cfg.num_images > 0 and images_processed >= cfg.num_images:
            logger.info(f"Stopping after {images_processed} images")
            break

        original_batch = data["image"]  # [B, C, H, W] in [0, 1]
        B = original_batch.shape[0]

        # Transmit
        start_time = time.time()
        transmitted, tx_times = transmitter(original_batch)
        elapsed_transmit_time = time.time() - start_time

        orig_size = channel.calculate_size_KB(original_batch)
        transmitted, trans_sizes = channel(transmitted)

        # Receive
        start_time = time.time()
        recon_tensor, rx_times, rx_data = receiver(transmitted)  # [B, C, H, W]
        elapsed_receive_time = time.time() - start_time

        for b in range(B):
            all_orig_images.append(original_batch[b].cpu())
            all_recon_images.append(recon_tensor[b].cpu())

        if cfg.save_images:
            os.makedirs("results", exist_ok=True)
            for b in range(B):
                img_idx = images_processed + b
                TF.to_pil_image(recon_tensor[b].cpu()).save(f"results/recon_{img_idx:03d}.png")

        # Token / bit error rates
        tx_tokens, tx_bits = transmitted["tx_tokens"], transmitted["tx_bits"]
        rx_tokens, rx_bits = rx_data["rx_tokens"], rx_data["rx_bits"]
        L_tok = min(len(tx_tokens), len(rx_tokens))
        L_bit = min(len(tx_bits), len(rx_bits))
        ter = float(np.mean(tx_tokens[:L_tok] != rx_tokens[:L_tok]))
        ber = float(np.mean(tx_bits[:L_bit] != rx_bits[:L_bit]))
        all_token_correct.append(tx_tokens[:L_tok] == rx_tokens[:L_tok])

        # Batch-averaged image metrics
        metrics = calculate_all_metrics(original_batch, recon_tensor)
        metrics.update(
            {
                "ter": ter,
                "ber": ber,
                "tx_time": elapsed_transmit_time,
                "rx_time": elapsed_receive_time,
                "compression_ratio": trans_sizes["total"] / orig_size,
            }
        )
        metrics.update({f"tx_{k}_time": v for k, v in tx_times.items()})
        metrics.update({f"rx_{k}_time": v for k, v in rx_times.items()})
        all_metrics.append(metrics)

        metrics_log = {k: f"{v:.4g}" for k, v in metrics.items()}
        pbar.set_postfix(metrics_log)
        logger.info(f"Metrics batch {batch_idx:03d}: {metrics_log}")
        wandb.log(metrics)

        images_processed += B

    # 6. Averages over batches
    np.save("token_correct.npy", np.concatenate(all_token_correct))

    avg_metrics = {"snr": cfg.channel.snr}
    avg_metrics.update({k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]})
    avg_metrics["fid"] = calculate_fid(torch.stack(all_orig_images), torch.stack(all_recon_images))
    avg_metrics["temperature"] = cfg.receiver.get("temperature", 1.0)
    avg_metrics["window_size"] = cfg.emu3.window_size if use_llm else 0
    avg_metrics["use_llm"] = use_llm
    avg_metrics["use_ldpc"] = cfg.channel.get("use_ldpc", False)
    avg_metrics["beam_width"] = cfg.receiver.get("beam_width", 1)
    avg_metrics["dataset"] = cfg.dataset.name
    avg_metrics["resolution"] = cfg.dataset.resolution
    pd.DataFrame([avg_metrics]).to_csv("avg_metrics.csv", index=False)
    logger.info("Metrics saved to avg_metrics.csv")

    for name, value in avg_metrics.items():
        wandb.log({f"avg_{name}": value})
        fmt = f"{value:.6g}" if isinstance(value, (int, float)) else str(value)
        logger.info(f"Average {name}: {fmt}")

    wandb.finish()


if __name__ == "__main__":
    main()
