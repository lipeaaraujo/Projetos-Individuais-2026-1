# src/data/dataset.py
"""
PyTorch Dataset and DataLoader utilities for YOLOS fine-tuning.

Labels are returned in DETR/YOLOS format: each sample's target is a dict
with ``class_labels`` (LongTensor) and ``boxes`` (FloatTensor, normalized
[cx, cy, w, h] in [0, 1]).  The custom ``collate_fn`` keeps labels as a
list of dicts because each image has a different number of objects.
"""
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import YolosImageProcessor

from src.data.preprocess import preprocess_image


class AstroDetectionDataset(Dataset):
    """
    Dataset for YOLOS fine-tuning on SDSS astronomical images.

    Each item is a ``(pixel_values, target)`` pair where:
    - ``pixel_values``: FloatTensor of shape ``(3, H, W)`` produced by
      ``YolosImageProcessor`` (resized, normalized).
    - ``target``: dict with keys ``class_labels`` (LongTensor) and
      ``boxes`` (FloatTensor ``[N, 4]`` in normalized [cx, cy, w, h] format).

    Parameters
    ----------
    image_dir:
        Directory containing the raw JPEG cutout images.
    metadata:
        List of annotation dicts as produced by ``build_annotated_dataset``::

            [{"image_file": "field_0000.jpg",
              "objects": [{"bbox": [cx, cy, w, h], "category_id": int}, ...]},
             ...]
    processor:
        A ``YolosImageProcessor`` instance used for image preprocessing.
    augment:
        If True, apply random flips and 90° rotations (safe for sky images).
    """

    def __init__(
        self,
        image_dir: Path,
        metadata: list[dict],
        processor: YolosImageProcessor,
        augment: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.metadata = metadata
        self.processor = processor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        entry = self.metadata[idx]
        img = Image.open(self.image_dir / entry["image_file"]).convert("RGB")
        img = preprocess_image(img)  # arcsinh stretch

        objects = entry["objects"]
        boxes = [o["bbox"] for o in objects]    # list of [cx, cy, w, h] normalized
        labels = [o["category_id"] for o in objects]

        if self.augment and objects:
            img, boxes = _random_augment(img, boxes)

        encoding = self.processor(images=img, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        target = {
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }
        return pixel_values, target


def collate_fn(batch: list[tuple]) -> tuple[torch.Tensor, list[dict]]:
    """
    Collate a list of (pixel_values, target) samples into a batch.

    ``pixel_values`` are stacked into a single tensor.
    ``targets`` remain as a list of dicts because each image can have a
    different number of objects — stacking is not possible.
    """
    pixel_values = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return pixel_values, targets


def load_annotations(annotations_path: Path) -> list[dict]:
    """Load the annotations.json file written by build_annotated_dataset."""
    return json.loads(annotations_path.read_text())


def train_val_split(
    metadata: list[dict],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split annotation metadata into train and validation sets at the
    image level (not object level) to avoid data leakage.
    """
    indices = list(range(len(metadata)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    val_idx = set(indices[:n_val])
    train = [metadata[i] for i in range(len(metadata)) if i not in val_idx]
    val = [metadata[i] for i in range(len(metadata)) if i in val_idx]
    return train, val


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _flip_boxes_h(boxes: list[list[float]]) -> list[list[float]]:
    """Horizontal flip: cx -> 1 - cx."""
    return [[1.0 - cx, cy, w, h] for cx, cy, w, h in boxes]


def _flip_boxes_v(boxes: list[list[float]]) -> list[list[float]]:
    """Vertical flip: cy -> 1 - cy."""
    return [[cx, 1.0 - cy, w, h] for cx, cy, w, h in boxes]


def _rot90_boxes(boxes: list[list[float]]) -> list[list[float]]:
    """90° counter-clockwise rotation: (cx, cy, w, h) -> (cy, 1-cx, h, w)."""
    return [[cy, 1.0 - cx, h, w] for cx, cy, w, h in boxes]


def _random_augment(
    img: Image.Image,
    boxes: list[list[float]],
) -> tuple[Image.Image, list[list[float]]]:
    """Apply random flips and 90° rotations. Sky images have no preferred orientation."""
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = _flip_boxes_h(boxes)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        boxes = _flip_boxes_v(boxes)
    k = random.randint(0, 3)
    for _ in range(k):
        img = img.rotate(90, expand=False)
        boxes = _rot90_boxes(boxes)
    return img, boxes
