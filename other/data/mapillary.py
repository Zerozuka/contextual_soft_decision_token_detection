import os
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class MapillaryDataset(Dataset):
    """Dataset for one split of Mapillary Vistas.

    Directory layout:
        <root>/training/images/*.jpg
        <root>/validation/images/*.jpg
        <root>/testing/images/*.jpg
    """

    SPLITS = {"train": "training", "val": "validation", "test": "testing"}

    def __init__(
        self,
        root: str,
        split: str = "val",
        transform: Optional[Callable] = None,
    ) -> None:
        split_dir = self.SPLITS.get(split)
        if split_dir is None:
            raise ValueError(f"split must be one of {list(self.SPLITS)}, got '{split}'")

        images_dir = Path(os.path.expanduser(root)) / split_dir / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        self.paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img}
