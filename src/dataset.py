"""Dataset definitions, preprocessing pipelines, data augmentations, and PyTorch DataLoaders.
"""

import os
from typing import Tuple, List, Dict, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Compose data augmentation and normalization pipeline for model training.

    Preserves critical morphological bird features while providing regularization
    via horizontal flips, mild rotations, and scaled crops.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Compose deterministic preprocessing pipeline for validation, testing, and inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class BirdDataset(Dataset):
    """PyTorch Dataset for Bird Species image samples.

    Supports loading either from a list of (image_path, label_idx) tuples or
    by scanning a split directory containing subdirectories named by species.
    Gracefully ensures all loaded images are converted to 3-channel RGB.
    """

    def __init__(
        self,
        samples: Optional[List[Tuple[str, int]]] = None,
        split_dir: Optional[str] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None
    ):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = class_to_idx or {}

        if samples is not None:
            self.samples = samples
            if not self.class_to_idx:
                unique_labels = sorted(list({s[1] for s in samples}))
                self.class_to_idx = {str(lbl): lbl for lbl in unique_labels}
        elif split_dir is not None:
            if not os.path.exists(split_dir):
                raise FileNotFoundError(f"Split directory not found: {split_dir}")

            subdirs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
            if not self.class_to_idx:
                self.class_to_idx = {sp: idx for idx, sp in enumerate(subdirs)}

            for sp in subdirs:
                if sp not in self.class_to_idx:
                    continue
                label_idx = self.class_to_idx[sp]
                sp_path = os.path.join(split_dir, sp)
                for fname in sorted(os.listdir(sp_path)):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        self.samples.append((os.path.join(sp_path, fname), label_idx))
        else:
            raise ValueError("Either 'samples' or 'split_dir' must be provided.")

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception as e:
            # Provide informative error on corrupted file
            raise RuntimeError(f"Error loading image '{img_path}': {e}")

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        return img_tensor, label


def create_dataloaders(
    processed_dir: str,
    config: dict
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]:
    """Build train, validation, and test PyTorch DataLoaders with safe worker configurations.

    Returns:
        (train_loader, val_loader, test_loader, class_names, class_to_idx)
    """
    image_size = config["dataset"].get("image_size", 224)
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["training"].get("num_workers", 0)  # 0 is safe on Windows
    selected_species = config["dataset"]["selected_species"]
    class_to_idx = {sp: idx for idx, sp in enumerate(selected_species)}

    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")
    test_dir = os.path.join(processed_dir, "test")

    train_dataset = BirdDataset(
        split_dir=train_dir,
        class_to_idx=class_to_idx,
        transform=get_train_transforms(image_size)
    )
    val_dataset = BirdDataset(
        split_dir=val_dir,
        class_to_idx=class_to_idx,
        transform=get_eval_transforms(image_size)
    )
    test_dataset = BirdDataset(
        split_dir=test_dir,
        class_to_idx=class_to_idx,
        transform=get_eval_transforms(image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, selected_species, class_to_idx
