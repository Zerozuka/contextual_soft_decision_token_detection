"""Test dataset loading."""

import os
from typing import Callable, Optional

import datasets
from omegaconf import DictConfig


def get_test_dataset(cfg: DictConfig, transform: Optional[Callable] = None) -> datasets.Dataset:
    """Return the FFHQ test images as a Hugging Face ``imagefolder`` dataset.

    ``cfg.dataset.path`` must point to a directory of FFHQ images (the paper
    uses the 512x512 version). Images are read in file-name order.

    Args:
        cfg: Hydra config.
        transform: optional PIL.Image -> Tensor function applied to the "image" field.
    """
    if cfg.dataset.name != "FFHQ":
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name} (only FFHQ is supported)")

    path = os.path.expanduser(cfg.dataset.get("path") or "~/datasets/ffhq_512")
    dataset = datasets.load_dataset("imagefolder", data_dir=path)["train"]

    if transform is not None:
        dataset.set_transform(
            lambda batch: {"image": [transform(img.convert("RGB")) for img in batch["image"]]}
        )
    return dataset
