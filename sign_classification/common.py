"""Dataset, preprocessing and frozen CLIP encoder shared by train/predict."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"不支持的图片格式: {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"路径不存在: {path}")
    return sorted(p for p in path.rglob("*") if p.is_file()
                  and p.suffix.lower() in IMAGE_EXTENSIONS)


def scan_dataset(root: Path, classes: list[str] | None = None):
    if not root.is_dir():
        raise ValueError(f"训练集目录不存在: {root}")
    found = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if classes is not None and set(found) != set(classes):
        raise ValueError(f"验证集类别必须与训练集一致，训练集={classes}，验证集={found}")
    classes = found if classes is None else classes
    if len(classes) < 2:
        raise ValueError("至少需要两个类别目录")
    paths, labels = [], []
    for label, name in enumerate(classes):
        files = image_paths(root / name)
        if not files:
            raise ValueError(f"类别目录没有图片: {root / name}")
        paths.extend(files)
        labels.extend([label] * len(files))
    return classes, paths, np.asarray(labels, dtype=np.int64)


def split_indices(labels: np.ndarray, val_ratio: float, seed: int):
    if not 0 <= val_ratio < 1:
        raise ValueError("val-ratio 必须在 [0, 1) 内")
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[int(label)].append(i)
    rng = random.Random(seed)
    train, val = [], []
    for label, indices in sorted(groups.items()):
        rng.shuffle(indices)
        if val_ratio and len(indices) < 2:
            raise ValueError(f"类别索引 {label} 只有一张图片，无法划分验证集；补充图片或用 --val-ratio 0")
        count = min(len(indices) - 1, max(1, round(len(indices) * val_ratio))) if val_ratio else 0
        val.extend(indices[:count])
        train.extend(indices[count:])
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(val), dtype=np.int64)


def load_image(path: Path) -> Image.Image:
    """White alpha compositing and square padding preserve the whole sign."""
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
    side = max(rgb.size)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    return canvas


class ClipEncoder:
    def __init__(self, model_name: str, pretrained: str, device: str, batch_size: int):
        # Lazy imports allow dataset checks and head tests without downloading CLIP.
        import open_clip
        import torch

        if batch_size < 1:
            raise ValueError("batch-size 必须大于 0")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        print(f"CLIP device={self.device}, model={model_name}", flush=True)

    def encode(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            raise ValueError("没有可提取特征的图片")
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                batch_paths = paths[start:start + self.batch_size]
                images = []
                for path in batch_paths:
                    try:
                        images.append(self.preprocess(load_image(path)))
                    except Exception as exc:
                        raise ValueError(f"读取图片失败: {path}: {exc}") from exc
                batch = self.torch.stack(images).to(self.device)
                features = self.model.encode_image(batch).float()
                features = self.torch.nn.functional.normalize(features, dim=-1)
                batches.append(features.cpu().numpy())
                print(f"提取特征 {min(start + self.batch_size, len(paths))}/{len(paths)}", flush=True)
        return np.concatenate(batches)
