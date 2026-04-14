from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from entmax import entmax_bisect
from torch import nn

from image_pipeline import (
    DatasetResult,
    FoldResult,
    ImageExperimentConfig,
    ImageRecord,
    ImageSoftClassifier,
    NormalizationStats,
    RegularizerDare,
    atomic_torch_save,
    build_transform,
    cleanup_training_state,
    export_prediction_frame,
    load_dataset_result,
    load_fold_result,
    load_image_dataset,
    load_member_summary_if_complete,
    load_prediction_frame,
    make_dataloader,
    mean_cross_entropy,
    resolve_normalization_stats,
    run_epoch,
    sanitize_path_token,
    serialize_config_for_run_name,
    set_seed,
    split_train_validation_records,
    write_json,
    save_results_summary
)


@dataclass(frozen=True, slots=True)
class EntmaxImageExperimentConfig:
    data_root: Path
    output_root: Path
    encoder_name: str
    pretrained: bool = True
    freeze_encoder: bool = True
    ensemble_size: int = 5
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_size: float = 0.1
    early_stopping_patience: int = 4
    num_workers: int = 4
    device: str = "cpu"
    seed: int = 42
    folds: list[str] | None = None
    normalization: str = "imagenet"
    augmentation: str = "basic"
    classifier_dropout: float = 0.0
    lambda_reg: float = 0.0
    amp: bool = True
    entmax_alpha: float = 1.5


class SoftTargetEntmaxBisectLoss(nn.Module):
    def __init__(self, alpha: float = 1.5, n_iter: int = 50):
        super().__init__()
        self.alpha = alpha
        self.n_iter = n_iter

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = entmax_bisect(
            logits,
            alpha=self.alpha,
            dim=1,
            n_iter=self.n_iter,
            ensure_sum_one=True,
        )
        omega = (1 - (probabilities**self.alpha).sum(dim=1)) / (
            self.alpha * (self.alpha - 1)
        )
        return (omega + ((probabilities - targets) * logits).sum(dim=1)).mean()


def serialize_config(config: EntmaxImageExperimentConfig) -> dict[str, Any]:
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
        "augmentation": config.augmentation,
        "classifier_dropout": config.classifier_dropout,
        "lambda_reg": config.lambda_reg,
        "amp": config.amp,
        "entmax_alpha": config.entmax_alpha,
    }


def serialize_for_run(config: EntmaxImageExperimentConfig, datasets: list[str]) -> dict[str, Any]:
    return {"datasets": datasets, "config": serialize_config(config)}


def build_run_name(config: EntmaxImageExperimentConfig) -> str:
    mode = "finetune" if not config.freeze_encoder else "linear"
    pretrained = "pretrained" if config.pretrained else "scratch"
    fold_part = sanitize_path_token("-".join(config.folds or ["all-folds"]))
    normalization_part = (
        "" if config.normalization == "imagenet" else f"_norm-{config.normalization}"
    )
    augmentation_part = (
        "" if config.augmentation == "basic" else f"_aug-{config.augmentation}"
    )
    regularizer_part = (
        "" if config.lambda_reg <= 0 else f"_reg-{sanitize_path_token(f'{config.lambda_reg:g}')}"
    )
    entmax_part = f"_entmax-{sanitize_path_token(f'{config.entmax_alpha:g}')}"
    readable = (
        f"{config.encoder_name}_entmax_{mode}_{pretrained}"
        f"_ens{config.ensemble_size}"
        f"_ep{config.epochs}"
        f"_bs{config.batch_size}"
        f"_seed{config.seed}"
        f"_{fold_part}"
        f"{'_amp' if config.amp else '_fp32'}"
        f"{regularizer_part}"
        f"{entmax_part}"
        f"{normalization_part}"
        f"{augmentation_part}"
    )
    fingerprint = hashlib.sha256(
        json.dumps(serialize_config_for_run_name(_as_image_config(config)), sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return f"{sanitize_path_token(readable)}_{fingerprint}"


def _as_image_config(config: EntmaxImageExperimentConfig) -> ImageExperimentConfig:
    return ImageExperimentConfig(
        data_root=config.data_root,
        output_root=config.output_root,
        encoder_name=config.encoder_name,
        pretrained=config.pretrained,
        freeze_encoder=config.freeze_encoder,
        ensemble_size=config.ensemble_size,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        validation_size=config.validation_size,
        early_stopping_patience=config.early_stopping_patience,
        num_workers=config.num_workers,
        device=config.device,
        seed=config.seed,
        folds=config.folds,
        normalization=config.normalization,
        augmentation=config.augmentation,
        classifier_dropout=config.classifier_dropout,
        lambda_reg=config.lambda_reg,
        amp=config.amp,
    )


def run_dataset_experiment(dataset_name: str, config: EntmaxImageExperimentConfig) -> DatasetResult:
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
            config=_as_image_config(config),
        )

        train_records = [record for record in records if record.fold != test_fold]
        test_records = [record for record in records if record.fold == test_fold]
        train_records, val_records = split_train_validation_records(
            train_records, validation_size=config.validation_size, seed=fold_seed
        )

        member_probabilities: list[np.ndarray] = []
        member_losses: list[float] = []
        member_prediction_files: list[str] = []
        model_files: list[str] = []

        for member_index in range(config.ensemble_size):
            probabilities, targets, member_summary, image_paths = train_single_member(
                member_index=member_index,
                fold_seed=fold_seed,
                fold_dir=fold_dir,
                test_fold=test_fold,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                class_names=class_names,
                dataset_name=dataset_name,
                config=config,
                normalization_stats=normalization_stats,
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

        fold_result = FoldResult(
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
        fold_results.append(fold_result)
        write_json(fold_summary_path, fold_result.to_dict())

    result = DatasetResult(
        dataset_name=dataset_name,
        encoder_name=config.encoder_name,
        class_names=class_names,
        folds=fold_results,
        mean_member_cross_entropy=float(
            np.mean([loss for fold in fold_results for loss in fold.member_cross_entropies])
        ),
        mean_ensemble_cross_entropy=float(
            np.mean([fold.ensemble_cross_entropy for fold in fold_results])
        ),
        config=serialize_config(config),
    )
    write_json(summary_path, result.to_dict())
    return result


def train_single_member(
    *,
    member_index: int,
    fold_seed: int,
    fold_dir: Path,
    test_fold: str,
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    test_records: list[ImageRecord],
    class_names: list[str],
    dataset_name: str,
    config: EntmaxImageExperimentConfig,
    normalization_stats: NormalizationStats,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    member_seed = fold_seed + 97 * member_index
    member_dir = fold_dir / f"member_{member_index:02d}"
    member_dir.mkdir(parents=True, exist_ok=True)
    member_summary = load_member_summary_if_complete(member_dir)

    if member_summary is None:
        model, history = train_single_model(
            train_records=train_records,
            val_records=val_records,
            num_classes=len(class_names),
            dataset_name=dataset_name,
            config=config,
            seed=member_seed,
            checkpoint_dir=member_dir,
            normalization_stats=normalization_stats,
        )
        probabilities, targets, image_paths = predict_records(
            model=model,
            records=test_records,
            dataset_name=dataset_name,
            config=config,
            normalization_stats=normalization_stats,
        )
        member_loss = mean_cross_entropy(targets, probabilities)
        model_path = member_dir / "model.pt"
        atomic_torch_save(model.state_dict(), model_path)
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
            Path(member_summary["prediction_file"]), class_names=class_names
        )

    return probabilities, targets, member_summary, image_paths


def train_single_model(
    *,
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    num_classes: int,
    dataset_name: str,
    config: EntmaxImageExperimentConfig,
    seed: int,
    checkpoint_dir: Path,
    normalization_stats: NormalizationStats,
) -> tuple[nn.Module, dict[str, list[float]]]:
    set_seed(seed)
    base_config = _as_image_config(config)
    train_loader = make_dataloader(
        data_root=config.data_root,
        records=train_records,
        transform=build_transform(base_config, dataset_name=dataset_name, train=True, normalization_stats=normalization_stats),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )
    val_loader = make_dataloader(
        data_root=config.data_root,
        records=val_records,
        transform=build_transform(base_config, dataset_name=dataset_name, train=False, normalization_stats=normalization_stats),
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
    criterion = SoftTargetEntmaxBisectLoss(alpha=config.entmax_alpha)
    use_amp = config.amp and config.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    regularizer = RegularizerDare(lambda_reg=config.lambda_reg) if config.lambda_reg > 0 else None

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = math.inf
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}
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

    for epoch_index in range(start_epoch, config.epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            config.device,
            optimizer,
            scaler=scaler,
            regularizer=regularizer,
            use_amp=use_amp,
            epoch_index=epoch_index,
        )
        val_loss = run_epoch(model, val_loader, criterion, config.device, use_amp=use_amp)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp else None,
                "best_state": best_state,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "completed_epochs": epoch_index + 1,
                "history": history,
            },
            checkpoint_path,
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_records(
    *,
    model: nn.Module,
    records: list[ImageRecord],
    dataset_name: str,
    config: EntmaxImageExperimentConfig,
    normalization_stats: NormalizationStats,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = make_dataloader(
        data_root=config.data_root,
        records=records,
        transform=build_transform(_as_image_config(config), dataset_name=dataset_name, train=False, normalization_stats=normalization_stats),
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
            probabilities = (
                entmax_bisect(logits.float(), alpha=config.entmax_alpha, dim=1, n_iter=50, ensure_sum_one=True)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            all_probabilities.append(probabilities)
            all_targets.append(targets.numpy())
            all_paths.extend(image_paths)
    return np.concatenate(all_probabilities, axis=0), np.concatenate(all_targets, axis=0), all_paths
