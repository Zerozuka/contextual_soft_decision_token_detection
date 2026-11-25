"""Image preprocessing."""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


def resize_and_crop_to_tensor(img: Image.Image, target_image_size: int) -> torch.Tensor:
    """Resize the shorter side to ``target_image_size`` (Lanczos), center-crop to a
    square and convert to a [C, H, W] tensor in [0, 1]."""
    if target_image_size is None:
        raise ValueError("target_image_size is None")

    s = min(img.size)
    if s < target_image_size:
        raise ValueError(f"Image is smaller than the target size: min side {s} < {target_image_size}")

    r = target_image_size / s
    new_size = (round(r * img.size[1]), round(r * img.size[0]))  # (H, W), aspect ratio preserved
    img = TF.resize(img, new_size, interpolation=T.InterpolationMode.LANCZOS)
    img = TF.center_crop(img, output_size=[target_image_size, target_image_size])
    return T.ToTensor()(img)
