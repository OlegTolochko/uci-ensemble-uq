from __future__ import annotations

from typing import Any

from PIL import Image
from torchvision import transforms


AUGMENTATION_MODES = ("none", "basic", "dcic", "dcic_auto")
DCIC_NO_AUG_DATASETS = {"CIFAR10H", "QualityMRI"}


class RandomQuarterTurnChoice:
    """Rotate by one of {0, 90, 180, 270} degrees."""

    def __init__(self):
        self.choice = transforms.RandomChoice(
            [
                transforms.Lambda(lambda image: image),
                transforms.Lambda(lambda image: image.rotate(90)),
                transforms.Lambda(lambda image: image.rotate(180)),
                transforms.Lambda(lambda image: image.rotate(270)),
            ]
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.choice(image)


def resolve_augmentation_mode(mode: str, dataset_name: str) -> str:
    if mode == "dcic_auto":
        if dataset_name in DCIC_NO_AUG_DATASETS:
            return "none"
        return "dcic"
    return mode


def build_train_augmentation_steps(
    *,
    mode: str,
    dataset_name: str,
    input_size: int,
) -> list[Any]:
    resolved_mode = resolve_augmentation_mode(mode, dataset_name)
    if resolved_mode == "none":
        return []

    if resolved_mode == "basic":
        return [transforms.RandomHorizontalFlip()]

    if resolved_mode == "dcic":
        return [
            transforms.RandomApply(
                [
                    transforms.Compose(
                        [
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                        ]
                    )
                ],
                p=0.5,
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=(0.5, 1.5),
                saturation=(0.6, 1.6),
                hue=0.08,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomResizedCrop(
                        input_size,
                        scale=(0.8, 1.0),
                        ratio=(1.0, 1.0),
                    )
                ],
                p=0.3,
            ),
            transforms.RandomApply([RandomQuarterTurnChoice()], p=0.5),
        ]

    raise ValueError(f"Unsupported augmentation mode: {mode}")
