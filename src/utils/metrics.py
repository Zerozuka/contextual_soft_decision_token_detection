"""Image-quality metrics: PSNR, SSIM, LPIPS, CLIP similarity and FID."""

from typing import Optional

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from loguru import logger
from torchmetrics.functional import peak_signal_noise_ratio as psnr_func
from torchmetrics.functional import structural_similarity_index_measure as ssim_func
from transformers import CLIPModel, CLIPProcessor

try:
    from torchmetrics.image.fid import FrechetInceptionDistance

    _FID_AVAILABLE = True
except Exception:  # torch-fidelity missing
    _FID_AVAILABLE = False

CLIP_MODEL = "openai/clip-vit-base-patch16"


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class MetricsEvaluator(nn.Module):
    """Holds the networks used by the model-based metrics (LPIPS-AlexNet, CLIP ViT-B/16)."""

    def __init__(self, device: str):
        super().__init__()
        self.device = torch.device(device)
        self.lpips_fn = lpips.LPIPS(net="alex").to(self.device).eval()
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

        # Half precision for CLIP on GPU
        if self.device.type == "cuda":
            self.clip.half()
            self.model_dtype = torch.float16
        else:
            self.model_dtype = torch.float32

    @torch.no_grad()
    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """L2-normalized CLIP image embeddings of [B, C, H, W] images in [0, 1]."""
        pil_imgs = [TF.to_pil_image(img.cpu()) for img in images]
        inputs = self.clip_processor(images=pil_imgs, return_tensors="pt").to(self.device)
        if self.device.type == "cuda":
            inputs = {k: v.to(self.model_dtype) if v.dtype == torch.float32 else v for k, v in inputs.items()}
        image_features = self.clip.get_image_features(**inputs)
        return F.normalize(image_features, p=2, dim=-1)


_EVALUATOR: Optional[MetricsEvaluator] = None


def get_evaluator(device: str) -> MetricsEvaluator:
    """Return a cached evaluator so that the networks are loaded only once."""
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = MetricsEvaluator(device)
    return _EVALUATOR


@torch.no_grad()
def calculate_all_metrics(original: torch.Tensor, reconstructed: torch.Tensor, device: Optional[str] = None) -> dict:
    """Batch-averaged PSNR, SSIM, LPIPS and CLIP cosine similarity.

    Args:
        original: [B, C, H, W] tensor in [0, 1].
        reconstructed: [B, C, H, W] tensor in [0, 1].
        device: computation device (default: CUDA if available, else CPU).
    """
    device = device or default_device()
    evaluator = get_evaluator(device)
    orig = original.to(device)
    recon = reconstructed.to(device)

    results = {
        "psnr": psnr_func(recon, orig, data_range=1.0).item(),
        "ssim": ssim_func(recon, orig, data_range=1.0).item(),
        "lpips": evaluator.lpips_fn((orig * 2) - 1, (recon * 2) - 1).mean().item(),
    }
    feat_orig = evaluator.extract_image_features(orig)
    feat_recon = evaluator.extract_image_features(recon)
    results["clip_similarity"] = torch.sum(feat_orig * feat_recon, dim=-1).mean().item()
    return results


@torch.no_grad()
def calculate_fid(original: torch.Tensor, reconstructed: torch.Tensor, device: Optional[str] = None) -> float:
    """FID between two image sets ([N, C, H, W] in [0, 1]).

    FID is unreliable for small N; it is reported but not used in the paper.
    """
    if not _FID_AVAILABLE:
        logger.warning("torch-fidelity is not installed; FID is skipped.")
        return float("nan")

    device = device or default_device()
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    def to_uint8(x):
        return (x * 255).clamp(0, 255).to(torch.uint8)

    fid_metric.update(to_uint8(original).to(device), real=True)
    fid_metric.update(to_uint8(reconstructed).to(device), real=False)
    return fid_metric.compute().item()
