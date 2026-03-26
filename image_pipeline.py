"""Image pipeline for soft-label classification.

Pipeline:
1. Read `annotations.json`.
2. For each image, count how many annotators picked each class.
3. Turn those vote counts into a probability distribution.
4. Use the dataset's predefined folds as the train/test split.
5. Split only the training folds again into train/validation.
6. Train a classifier with soft-target cross entropy.
7. Save predictions and model weights.

`ensemble_size` means: how many independently initialized models to train
for the same held-out test fold. It does not control how many folds are used.

Examples:
- `folds=None`, `ensemble_size=1` on a 5-fold dataset -> train 5 models total
    (one per held-out test fold).
- `folds=["fold1"]`, `ensemble_size=1` -> train exactly 1 model.
- `folds=["fold1"]`, `ensemble_size=5` -> train 5 models on the same split and
    average their probabilities.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import get_model, get_model_weights


torch.set_float32_matmul_precision("high")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMALIZATION_STATS_FILE = "normalization_stats.json"

DEFAULT_ENCODERS = {
    "resnet18": 224,
    "resnet50": 224,
    "wide_resnet50_2": 224,
    "resnext50_32x4d": 224,
    "densenet121": 224,
    "efficientnet_b0": 224,
    "convnext_tiny": 224,
    "vit_b_16": 224,
}


@dataclass(slots=True)
class ImageRecord:
    image_path: str
    fold: str  # e.g. `fold0`, `fold1`, etc.
    target_probs: np.ndarray  # e.g. [0.1, 0.8, 0.1]


@dataclass(slots=True)
class ImageExperimentConfig:
    data_root: Path = Path("data/image")
    output_root: Path = Path("out/image")
    encoder_name: str = "resnet18"
    pretrained: bool = True
    freeze_encoder: bool = True
    ensemble_size: int = 5
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_size: float = 0.1
    early_stopping_patience: int = 4
    num_workers: int = 4
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    seed: int = 42
    folds: list[str] | None = None
    normalization: str = "imagenet"
    classifier_dropout: float = 0.0
    amp: bool = True


@dataclass(slots=True)
class FoldResult:
    test_fold: str
    train_size: int
    val_size: int
    test_size: int
    member_cross_entropies: list[float]
    ensemble_cross_entropy: float
    prediction_file: str
    member_prediction_files: list[str]
    model_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_fold": self.test_fold,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "member_cross_entropies": self.member_cross_entropies,
            "ensemble_cross_entropy": self.ensemble_cross_entropy,
            "prediction_file": self.prediction_file,
            "member_prediction_files": self.member_prediction_files,
            "model_files": self.model_files,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FoldResult":
        return cls(
            test_fold=payload["test_fold"],
            train_size=payload["train_size"],
            val_size=payload["val_size"],
            test_size=payload["test_size"],
            member_cross_entropies=list(payload["member_cross_entropies"]),
            ensemble_cross_entropy=payload["ensemble_cross_entropy"],
            prediction_file=payload["prediction_file"],
            member_prediction_files=list(payload["member_prediction_files"]),
            model_files=list(payload["model_files"]),
        )


@dataclass(slots=True)
class DatasetResult:
    dataset_name: str
    encoder_name: str
    class_names: list[str]
    folds: list[FoldResult]
    mean_member_cross_entropy: float
    mean_ensemble_cross_entropy: float
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "encoder_name": self.encoder_name,
            "class_names": self.class_names,
            "mean_member_cross_entropy": self.mean_member_cross_entropy,
            "mean_ensemble_cross_entropy": self.mean_ensemble_cross_entropy,
            "config": self.config,
            "folds": [fold.to_dict() for fold in self.folds],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetResult":
        return cls(
            dataset_name=payload["dataset_name"],
            encoder_name=payload["encoder_name"],
            class_names=list(payload["class_names"]),
            folds=[FoldResult.from_dict(fold) for fold in payload["folds"]],
            mean_member_cross_entropy=payload["mean_member_cross_entropy"],
            mean_ensemble_cross_entropy=payload["mean_ensemble_cross_entropy"],
            config=dict(payload["config"]),
        )


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class SoftLabelImageDataset(Dataset):
    """Wrapper that loads images and returns soft targets"""

    def __init__(
        self,
        data_root: Path,
        records: list[ImageRecord],
        transform: transforms.Compose,
    ):
        self.data_root = data_root
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        image = Image.open(self.data_root / record.image_path).convert("RGB")
        target = torch.tensor(record.target_probs, dtype=torch.float32)
        return self.transform(image), target, record.image_path


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()


class ImageSoftClassifier(nn.Module):
    """Encoder with a replaceable classification head."""

    def __init__(
        self,
        encoder_name: str,
        num_classes: int,
        pretrained: bool,
        freeze_encoder: bool,
        classifier_dropout: float,
    ):
        super().__init__()

        weights = None
        if pretrained:
            weights = get_model_weights(encoder_name).DEFAULT

        self.encoder = get_model(encoder_name, weights=weights)
        in_features = replace_classification_head_with_identity(
            self.encoder, encoder_name
        )
        self.head = build_classifier_head(
            in_features=in_features,
            num_classes=num_classes,
            classifier_dropout=classifier_dropout,
        )

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        if features.ndim > 2: # if the encoder doesn't already pool to (batch_size, features)
            features = torch.flatten(features, start_dim=1)
        return self.head(features)


def list_available_encoders() -> list[str]:
    return sorted(DEFAULT_ENCODERS)


def build_classifier_head(
    *,
    in_features: int,
    num_classes: int,
    classifier_dropout: float,
) -> nn.Module:
    if classifier_dropout > 0:
        return nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(in_features, num_classes),
        )

    return nn.Linear(in_features, num_classes)


def discover_image_datasets(data_root: Path) -> list[str]:
    return sorted(path.name for path in data_root.iterdir() if path.is_dir())


def load_image_dataset(dataset_dir: Path) -> tuple[list[str], list[ImageRecord]]:
    """Load a dataset and convert annotator votes into soft labels.
    Example:
    - votes: `cat=8`, `dog=2`
    - target distribution: `[0.8, 0.2]`
    """

    with (dataset_dir / "annotations.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    annotation_groups = payload if isinstance(payload, list) else [payload]
    vote_counts_by_image: dict[str, Counter[str]] = defaultdict(Counter)
    class_names: set[str] = set()

    for group in annotation_groups:
        for annotation in group.get("annotations", []):
            image_path = annotation["image_path"]
            class_name = annotation["class_label"]
            vote_counts_by_image[image_path][class_name] += 1
            class_names.add(class_name)

    ordered_classes = sorted(class_names)
    records: list[ImageRecord] = []

    for image_path, class_counts in sorted(vote_counts_by_image.items()):
        total_votes = sum(class_counts.values())
        target_probs = np.array(
            [
                class_counts.get(class_name, 0) / total_votes
                for class_name in ordered_classes
            ],
            dtype=np.float32,
        )
        fold = extract_fold_name(image_path)
        records.append(
            ImageRecord(image_path=image_path, fold=fold, target_probs=target_probs)
        )

    return ordered_classes, records


def run_dataset_experiment(
    dataset_name: str,
    config: ImageExperimentConfig,
) -> DatasetResult:
    """Train and evaluate one dataset.

    Data splitting happens in two stages:
    1. Use one predefined fold as the held-out test set.
    2. Split the remaining folds into train and validation.
    """

    output_dir = config.output_root / dataset_name / config.encoder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    write_json(output_dir / "config.json", serialize_config(config))

    if summary_path.exists():
        return load_dataset_result(summary_path)

    class_names, records = load_image_dataset(config.data_root / dataset_name)

    fold_names = config.folds or sorted({record.fold for record in records})
    fold_results: list[FoldResult] = []

    for fold_offset, test_fold in enumerate(fold_names):
        fold_dir = output_dir / test_fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_summary_path = fold_dir / "summary.json"

        if fold_summary_path.exists():
            fold_results.append(load_fold_result(fold_summary_path))
            continue

        fold_seed = config.seed + fold_offset
        normalization_stats = resolve_normalization_stats(
            dataset_name=dataset_name,
            records=records,
            test_fold=test_fold,
            config=config,
        )

        train_records = [record for record in records if record.fold != test_fold]
        test_records = [record for record in records if record.fold == test_fold]

        train_records, val_records = split_train_validation_records(
            train_records,
            validation_size=config.validation_size,
            seed=fold_seed,
        )

        member_probabilities: list[np.ndarray] = []
        member_losses: list[float] = []
        member_prediction_files: list[str] = []
        model_files: list[str] = []

        for member_index in range(config.ensemble_size):
            member_seed = fold_seed + 97 * member_index
            member_dir = fold_dir / f"member_{member_index:02d}"
            member_dir.mkdir(parents=True, exist_ok=True)
            member_summary = load_member_summary_if_complete(member_dir)

            if member_summary is None:
                model, history = train_single_model(
                    train_records=train_records,
                    val_records=val_records,
                    num_classes=len(class_names),
                    config=config,
                    seed=member_seed,
                    checkpoint_dir=member_dir,
                    normalization_stats=normalization_stats,
                )

                probabilities, targets, image_paths = predict_records(
                    model=model,
                    records=test_records,
                    config=config,
                    normalization_stats=normalization_stats,
                )

                member_loss = mean_cross_entropy(targets, probabilities)
                model_path = member_dir / "model.pt"
                torch.save(model.state_dict(), model_path)

                member_prediction_path = member_dir / "predictions.csv"
                export_prediction_frame(
                    output_path=member_prediction_path,
                    image_paths=image_paths,
                    targets=targets,
                    probabilities=probabilities,
                    class_names=class_names,
                    fold_name=test_fold,
                    history=history,
                )
                member_summary = {
                    "member_index": member_index,
                    "member_seed": member_seed,
                    "cross_entropy": member_loss,
                    "model_file": str(model_path),
                    "prediction_file": str(member_prediction_path),
                    "history_file": str(member_prediction_path.with_name("history.json")),
                }
                write_json(member_dir / "summary.json", member_summary)
                cleanup_training_state(member_dir)
            else:
                probabilities, targets, image_paths = load_prediction_frame(
                    Path(member_summary["prediction_file"]),
                    class_names=class_names,
                )

            member_probabilities.append(probabilities)
            member_losses.append(float(member_summary["cross_entropy"]))
            member_prediction_files.append(member_summary["prediction_file"])
            model_files.append(member_summary["model_file"])

        ensemble_probabilities = np.mean(np.stack(member_probabilities, axis=0), axis=0)
        ensemble_loss = mean_cross_entropy(targets, ensemble_probabilities)

        ensemble_prediction_path = fold_dir / "ensemble_predictions.csv"
        export_prediction_frame(
            output_path=ensemble_prediction_path,
            image_paths=image_paths,
            targets=targets,
            probabilities=ensemble_probabilities,
            class_names=class_names,
            fold_name=test_fold,
            history=None,
        )

        fold_results.append(
            FoldResult(
                test_fold=test_fold,
                train_size=len(train_records),
                val_size=len(val_records),
                test_size=len(test_records),
                member_cross_entropies=member_losses,
                ensemble_cross_entropy=ensemble_loss,
                prediction_file=str(ensemble_prediction_path),
                member_prediction_files=member_prediction_files,
                model_files=model_files,
            )
        )
        write_json(fold_summary_path, fold_results[-1].to_dict())

    result = DatasetResult(
        dataset_name=dataset_name,
        encoder_name=config.encoder_name,
        class_names=class_names,
        folds=fold_results,
        mean_member_cross_entropy=float(
            np.mean(
                [loss for fold in fold_results for loss in fold.member_cross_entropies]
            )
        ),
        mean_ensemble_cross_entropy=float(
            np.mean([fold.ensemble_cross_entropy for fold in fold_results])
        ),
        config=serialize_config(config),
    )

    write_json(summary_path, result.to_dict())

    return result


def train_single_model(
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    num_classes: int,
    config: ImageExperimentConfig,
    seed: int,
    checkpoint_dir: Path,
    normalization_stats: NormalizationStats,
) -> tuple[nn.Module, dict[str, list[float]]]:
    set_seed(seed)

    train_loader = make_dataloader(
        data_root=config.data_root,
        records=train_records,
        transform=build_transform(config, train=True, normalization_stats=normalization_stats),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )
    val_loader = make_dataloader(
        data_root=config.data_root,
        records=val_records,
        transform=build_transform(config, train=False, normalization_stats=normalization_stats),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )

    model = ImageSoftClassifier(
        encoder_name=config.encoder_name,
        num_classes=num_classes,
        pretrained=config.pretrained,
        freeze_encoder=config.freeze_encoder,
        classifier_dropout=config.classifier_dropout,
    ).to(config.device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = SoftTargetCrossEntropy()
    use_amp = config.amp and config.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = math.inf
    epochs_without_improvement = 0
    history = {"train_cross_entropy": [], "val_cross_entropy": []}
    checkpoint_path = checkpoint_dir / "training_state.pt"
    start_epoch = 0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        best_state = checkpoint["best_state"]
        best_val_loss = checkpoint["best_val_loss"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        history = checkpoint["history"]
        start_epoch = checkpoint["completed_epochs"]
        restore_rng_state(checkpoint["rng_state"])

    for epoch_index in range(start_epoch, config.epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            config.device,
            optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            config.device,
            use_amp=use_amp,
        )

        history["train_cross_entropy"].append(train_loss)
        history["val_cross_entropy"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp else None,
                "best_state": best_state,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "completed_epochs": epoch_index + 1,
                "history": history,
                "rng_state": capture_rng_state(),
            },
            checkpoint_path,
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_records(
    model: nn.Module,
    records: list[ImageRecord],
    config: ImageExperimentConfig,
    normalization_stats: NormalizationStats,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = make_dataloader(
        data_root=config.data_root,
        records=records,
        transform=build_transform(config, train=False, normalization_stats=normalization_stats),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )

    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_paths: list[str] = []

    model.eval()
    with torch.no_grad():
        for images, targets, image_paths in loader:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=config.amp and config.device.startswith("cuda"),
            ):
                logits = model(images.to(config.device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            all_probabilities.append(probabilities)
            all_targets.append(targets.numpy())
            all_paths.extend(image_paths)

    return (
        np.concatenate(all_probabilities, axis=0),
        np.concatenate(all_targets, axis=0),
        all_paths,
    )


def split_train_validation_records(
    records: list[ImageRecord],
    validation_size: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split the non-test data into train and validation."""

    stratify = np.array([int(np.argmax(record.target_probs)) for record in records])

    train_indices, val_indices = train_test_split(
        np.arange(len(records)),
        test_size=validation_size,
        random_state=seed,
        stratify=stratify,
    )
    train_records = [records[index] for index in train_indices]
    val_records = [records[index] for index in val_indices]
    return train_records, val_records


def build_transform(
    config: ImageExperimentConfig,
    train: bool,
    normalization_stats: NormalizationStats,
) -> transforms.Compose:
    input_size = DEFAULT_ENCODERS[config.encoder_name]

    steps: list[Any] = []
    if train:
        steps.append(transforms.RandomHorizontalFlip())

    steps.append(transforms.Resize((input_size, input_size)))

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(normalization_stats.mean, normalization_stats.std),
        ]
    )
    return transforms.Compose(steps)


def make_dataloader(
    data_root: Path,
    records: list[ImageRecord],
    transform: transforms.Compose,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = SoftLabelImageDataset(data_root, records, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def export_prediction_frame(
    output_path: Path,
    image_paths: list[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    fold_name: str,
    history: dict[str, list[float]] | None,
):
    frame = pd.DataFrame(
        {
            "image_path": image_paths,
            "fold": fold_name,
            "cross_entropy": cross_entropy_per_sample(targets, probabilities),
            "target_entropy": entropy_per_sample(targets),
        }
    )

    for class_index, class_name in enumerate(class_names):
        frame[f"target::{class_name}"] = targets[:, class_index]
        frame[f"pred::{class_name}"] = probabilities[:, class_index]

    frame.to_csv(output_path, index=False)

    if history is not None:
        with output_path.with_name("history.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(history, handle, indent=2)


def mean_cross_entropy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(cross_entropy_per_sample(targets, probabilities).mean())


def cross_entropy_per_sample(
    targets: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    return -(targets * np.log(clipped)).sum(axis=1)


def entropy_per_sample(distributions: np.ndarray) -> np.ndarray:
    clipped = np.clip(distributions, 1e-8, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def save_results_summary(output_root: Path, results: list[DatasetResult]):
    output_root.mkdir(parents=True, exist_ok=True)
    existing_payload: list[dict[str, Any]] = []
    results_path = output_root / "results.json"
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            existing_payload = json.load(handle)

    merged_by_dataset = {
        item["dataset_name"]: item for item in existing_payload
    }
    for result in results:
        merged_by_dataset[result.dataset_name] = result.to_dict()

    payload = [
        merged_by_dataset[dataset_name]
        for dataset_name in sorted(merged_by_dataset)
    ]
    write_json(results_path, payload)


def cleanup_training_state(member_dir: Path) -> None:
    training_state_path = member_dir / "training_state.pt"
    if training_state_path.exists():
        training_state_path.unlink()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> float:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_items = 0

    with torch.set_grad_enabled(is_training):
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

    return total_loss / max(total_items, 1)


def replace_classification_head_with_identity(
    model: nn.Module, encoder_name: str
) -> int:
    """Remove the ImageNet classifier so we can attach our own output layer. (more may be added)"""

    if encoder_name in {"resnet18", "resnet50", "wide_resnet50_2", "resnext50_32x4d"}:
        in_features = int(model.fc.in_features)
        model.fc = nn.Identity()
        return in_features

    if encoder_name == "densenet121":
        in_features = int(model.classifier.in_features)
        model.classifier = nn.Identity()
        return in_features

    if encoder_name in {"efficientnet_b0", "convnext_tiny"}:
        in_features = int(model.classifier[-1].in_features)
        model.classifier = nn.Identity()
        return in_features

    if encoder_name == "vit_b_16":
        in_features = int(model.heads.head.in_features)
        model.heads = nn.Identity()
        return in_features

    raise ValueError(f"Unsupported encoder: {encoder_name}")


def extract_fold_name(image_path: str) -> str:
    parts = Path(image_path).parts
    return parts[1] if len(parts) > 1 else "fold0"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialize_config(config: ImageExperimentConfig) -> dict[str, Any]:
    return {
        "data_root": str(config.data_root),
        "output_root": str(config.output_root),
        "encoder_name": config.encoder_name,
        "pretrained": config.pretrained,
        "freeze_encoder": config.freeze_encoder,
        "ensemble_size": config.ensemble_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "validation_size": config.validation_size,
        "early_stopping_patience": config.early_stopping_patience,
        "num_workers": config.num_workers,
        "device": config.device,
        "seed": config.seed,
        "folds": config.folds,
        "normalization": config.normalization,
        "classifier_dropout": config.classifier_dropout,
        "amp": config.amp,
    }


def build_run_name(config: ImageExperimentConfig) -> str:
    mode = "finetune" if not config.freeze_encoder else "head-only"
    pretrained = "pretrained" if config.pretrained else "scratch"
    fold_part = "all-folds" if not config.folds else "-".join(sorted(config.folds))
    normalization_part = (
        ""
        if config.normalization == "imagenet"
        else f"_norm-{sanitize_path_token(config.normalization)}"
    )
    readable = (
        f"{config.encoder_name}_{mode}_{pretrained}"
        f"_ens{config.ensemble_size}"
        f"_ep{config.epochs}"
        f"_bs{config.batch_size}"
        f"_seed{config.seed}"
        f"_{fold_part}"
        f"{'_amp' if config.amp else '_fp32'}"
        f"{normalization_part}"
    )
    fingerprint = hashlib.sha256(
        json.dumps(serialize_config_for_run_name(config), sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return f"{sanitize_path_token(readable)}_{fingerprint}"


def serialize_config_for_run_name(config: ImageExperimentConfig) -> dict[str, Any]:
    payload = serialize_config(config)
    if config.normalization == "imagenet":
        payload.pop("normalization", None)
    return payload


def sanitize_path_token(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value
    ).strip("-")


def resolve_normalization_stats(
    dataset_name: str,
    records: list[ImageRecord],
    test_fold: str,
    config: ImageExperimentConfig,
) -> NormalizationStats:
    if config.normalization == "imagenet":
        return NormalizationStats(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if config.normalization != "dataset":
        raise ValueError(f"Unsupported normalization mode: {config.normalization}")

    dataset_dir = config.data_root / dataset_name
    stats_payload = load_or_compute_dataset_normalization_stats(
        dataset_dir=dataset_dir,
        records=records,
        data_root=config.data_root,
    )
    fold_stats = stats_payload["fold_stats"][test_fold]
    return NormalizationStats(
        mean=tuple(float(value) for value in fold_stats["mean"]),
        std=tuple(float(value) for value in fold_stats["std"]),
    )


def load_or_compute_dataset_normalization_stats(
    dataset_dir: Path,
    records: list[ImageRecord],
    data_root: Path,
) -> dict[str, Any]:
    stats_path = dataset_dir / NORMALIZATION_STATS_FILE
    fold_names = sorted({record.fold for record in records})

    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cached_folds = sorted(payload.get("fold_stats", {}).keys())
        if cached_folds == fold_names:
            return payload

    fold_stats: dict[str, dict[str, Any]] = {}
    for held_out_fold in fold_names:
        training_paths = [
            data_root / record.image_path
            for record in records
            if record.fold != held_out_fold
        ]
        mean, std = compute_channel_stats(training_paths)
        fold_stats[held_out_fold] = {
            "mean": list(mean),
            "std": list(std),
            "num_images": len(training_paths),
        }

    payload = {
        "mode": "dataset",
        "description": "Per-held-out-fold RGB channel stats computed from all non-held-out folds.",
        "fold_stats": fold_stats,
    }
    write_json(stats_path, payload)
    return payload


def compute_channel_stats(image_paths: list[Path]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for image_path in image_paths:
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        flat = image.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_squared_sum += np.square(flat).sum(axis=0)
        pixel_count += flat.shape[0]

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - np.square(mean)
    std = np.sqrt(np.clip(variance, a_min=1e-12, a_max=None))
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["cuda"]])


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_dataset_result(path: Path) -> DatasetResult:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return DatasetResult.from_dict(payload)


def load_fold_result(path: Path) -> FoldResult:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return FoldResult.from_dict(payload)


def load_member_summary_if_complete(member_dir: Path) -> dict[str, Any] | None:
    summary_path = member_dir / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required_files = [
        Path(payload["model_file"]),
        Path(payload["prediction_file"]),
        Path(payload["history_file"]),
    ]
    if all(path.exists() for path in required_files):
        return payload
    return None


def load_prediction_frame(
    prediction_path: Path,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(prediction_path)
    target_columns = [f"target::{class_name}" for class_name in class_names]
    prediction_columns = [f"pred::{class_name}" for class_name in class_names]
    return (
        frame[prediction_columns].to_numpy(dtype=np.float32),
        frame[target_columns].to_numpy(dtype=np.float32),
        frame["image_path"].tolist(),
    )
