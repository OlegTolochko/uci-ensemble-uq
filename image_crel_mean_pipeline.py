from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from image_pipeline import (
    DatasetResult,
    FoldResult,
    ImageExperimentConfig,
    ImageRecord,
    ImageSoftClassifier,
    NormalizationStats,
    SoftTargetCrossEntropy,
    atomic_torch_save,
    build_transform,
    capture_rng_state,
    cleanup_training_state,
    export_prediction_frame,
    load_dataset_result,
    load_fold_result,
    load_image_dataset,
    load_prediction_frame,
    make_dataloader,
    mean_cross_entropy,
    predict_records,
    resolve_normalization_stats,
    restore_rng_state,
    run_epoch,
    sanitize_path_token,
    save_results_summary,
    set_seed,
    split_train_validation_records,
    train_single_model,
    write_json,
)


@dataclass(slots=True)
class CreRLMeanImageExperimentConfig:
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    folds: list[str] | None = None
    normalization: str = "imagenet"
    augmentation: str = "basic"
    classifier_dropout: float = 0.0
    amp: bool = True
    alpha: float = 0.8
    tobias_strength: float = 100.0


def apply_tobias_initialization(
    *,
    model: ImageSoftClassifier,
    num_classes: int,
    member_index: int,
    tobias_strength: float,
) -> None:
    final_linear = model.head[-1] if isinstance(model.head, nn.Sequential) else model.head
    target_class = member_index % num_classes
    with torch.no_grad():
        final_linear.bias[target_class] = tobias_strength


def build_tau_schedule(*, alpha: float, ensemble_size: int) -> list[float]:
    if ensemble_size <= 1:
        return [1.0]
    return [float(value) for value in np.linspace(alpha, 1.0, num=ensemble_size)]


def run_dataset_experiment(
    dataset_name: str,
    config: CreRLMeanImageExperimentConfig,
) -> DatasetResult:
    output_dir = config.output_root / dataset_name / config.encoder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    write_json(output_dir / "config.json", serialize_config(config))

    if summary_path.exists():
        return load_dataset_result(summary_path)

    class_names, records = load_image_dataset(config.data_root / dataset_name)
    num_classes = len(class_names)
    standard_config = to_standard_config(config)
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
            config=standard_config,
        )
        train_records = [record for record in records if record.fold != test_fold]
        test_records = [record for record in records if record.fold == test_fold]
        train_records, val_records = split_train_validation_records(
            train_records,
            validation_size=config.validation_size,
            seed=fold_seed,
        )

        h_ml_dir = fold_dir / "h_ml"
        h_ml_dir.mkdir(parents=True, exist_ok=True)
        h_ml_train_loss_sum = train_or_load_h_ml(
            dataset_name=dataset_name,
            train_records=train_records,
            val_records=val_records,
            num_classes=num_classes,
            config=config,
            seed=fold_seed,
            checkpoint_dir=h_ml_dir,
            normalization_stats=normalization_stats,
        )
        h_ml_train_ce_mean = float(h_ml_train_loss_sum / max(len(train_records), 1))

        tau_values = build_tau_schedule(
            alpha=config.alpha,
            ensemble_size=config.ensemble_size,
        )

        member_probabilities: list[np.ndarray] = []
        member_losses: list[float] = []
        member_prediction_files: list[str] = []
        model_files: list[str] = []

        for member_index, tau in enumerate(tau_values):
            member_seed = fold_seed + 97 * member_index
            member_dir = fold_dir / f"member_{member_index:02d}"
            member_dir.mkdir(parents=True, exist_ok=True)
            member_summary_path = member_dir / "summary.json"
            member_summary = None
            if member_summary_path.exists():
                with member_summary_path.open("r", encoding="utf-8") as handle:
                    member_summary = json.load(handle)

            if member_summary is None:
                (
                    model,
                    history,
                    reached_epoch,
                    final_train_ce_mean,
                    final_relative_likelihood,
                ) = train_crel_mean_member_model(
                    dataset_name=dataset_name,
                    train_records=train_records,
                    num_classes=num_classes,
                    config=config,
                    seed=member_seed,
                    checkpoint_dir=member_dir,
                    normalization_stats=normalization_stats,
                    member_index=member_index,
                    tau=tau,
                    h_ml_train_ce_mean=h_ml_train_ce_mean,
                )
                probabilities, targets, image_paths = predict_records(
                    model=model,
                    records=test_records,
                    dataset_name=dataset_name,
                    config=standard_config,
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
                    "tau": float(tau),
                    "final_train_cross_entropy_mean": float(final_train_ce_mean),
                    "final_relative_likelihood": float(final_relative_likelihood),
                    "stopped_epoch": reached_epoch,
                }
                write_json(member_summary_path, member_summary)
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


def train_crel_mean_member_model(
    *,
    dataset_name: str,
    train_records: list[ImageRecord],
    num_classes: int,
    config: CreRLMeanImageExperimentConfig,
    seed: int,
    checkpoint_dir: Path,
    normalization_stats: NormalizationStats,
    member_index: int,
    tau: float,
    h_ml_train_ce_mean: float,
) -> tuple[nn.Module, dict[str, list[float]], int, float, float]:
    set_seed(seed)
    standard_config = to_standard_config(config)
    train_loader = make_dataloader(
        data_root=config.data_root,
        records=train_records,
        transform=build_transform(
            standard_config,
            dataset_name=dataset_name,
            train=True,
            normalization_stats=normalization_stats,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )
    eval_train_loader = make_dataloader(
        data_root=config.data_root,
        records=train_records,
        transform=build_transform(
            standard_config,
            dataset_name=dataset_name,
            train=False,
            normalization_stats=normalization_stats,
        ),
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
    apply_tobias_initialization(
        model=model,
        num_classes=num_classes,
        member_index=member_index,
        tobias_strength=config.tobias_strength,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = SoftTargetCrossEntropy()
    use_amp = config.amp and config.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_path = checkpoint_dir / "training_state.pt"

    initial_train_ce_mean = evaluate_soft_ce_mean_from_loader(
        model=model,
        loader=eval_train_loader,
        device=config.device,
        use_amp=use_amp,
    )
    history = {
        "train_cross_entropy": [],
        "train_cross_entropy_mean": [],
        "train_relative_likelihood": [],
        "tau": [float(tau)],
        "initial_train_cross_entropy_mean": [float(initial_train_ce_mean)],
    }
    start_epoch = 0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if use_amp:
            scaler.load_state_dict(checkpoint["scaler_state"])
        history = checkpoint["history"]
        start_epoch = checkpoint["completed_epochs"]
        restore_rng_state(checkpoint["rng_state"])

    final_train_ce_mean = math.inf
    final_relative_likelihood = 0.0
    stopped_epoch = start_epoch
    for epoch_index in range(start_epoch, config.epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            config.device,
            optimizer,
            scaler=scaler,
            use_amp=use_amp,
            epoch_index=epoch_index,
        )
        final_train_ce_mean = evaluate_soft_ce_mean_from_loader(
            model=model,
            loader=eval_train_loader,
            device=config.device,
            use_amp=use_amp,
        )
        final_relative_likelihood = float(math.exp(h_ml_train_ce_mean - final_train_ce_mean))
        history["train_cross_entropy"].append(train_loss)
        history["train_cross_entropy_mean"].append(final_train_ce_mean)
        history["train_relative_likelihood"].append(final_relative_likelihood)
        stopped_epoch = epoch_index + 1
        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp else None,
                "history": history,
                "completed_epochs": stopped_epoch,
                "rng_state": capture_rng_state(),
            },
            checkpoint_path,
        )
        if final_relative_likelihood >= float(tau):
            break

    model.eval()
    return (
        model,
        history,
        stopped_epoch,
        final_train_ce_mean,
        final_relative_likelihood,
    )


def train_or_load_h_ml(
    *,
    dataset_name: str,
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    num_classes: int,
    config: CreRLMeanImageExperimentConfig,
    seed: int,
    checkpoint_dir: Path,
    normalization_stats: NormalizationStats,
) -> float:
    summary_path = checkpoint_dir / "summary.json"
    model_path = checkpoint_dir / "model.pt"
    history_path = checkpoint_dir / "history.json"

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        return float(summary["train_loss_sum"])

    model, history = train_single_model(
        train_records=train_records,
        val_records=val_records,
        num_classes=num_classes,
        dataset_name=dataset_name,
        config=to_standard_config(config),
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        normalization_stats=normalization_stats,
    )
    train_loss_sum = evaluate_soft_ce_sum(
        model=model,
        records=train_records,
        dataset_name=dataset_name,
        config=config,
        normalization_stats=normalization_stats,
    )
    atomic_torch_save(model.state_dict(), model_path)
    write_json(history_path, history)
    write_json(
        summary_path,
        {
            "model_file": str(model_path),
            "history_file": str(history_path),
            "train_loss_sum": float(train_loss_sum),
            "train_loss_mean": float(train_loss_sum / max(len(train_records), 1)),
        },
    )
    cleanup_training_state(checkpoint_dir)
    return train_loss_sum


def evaluate_soft_ce(
    *,
    model: nn.Module,
    loader: Any,
    device: str,
    use_amp: bool,
) -> tuple[float, int]:
    criterion = SoftTargetCrossEntropy()
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
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
            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
    return total_loss, total_items


def evaluate_soft_ce_sum(
    *,
    model: nn.Module,
    records: list[ImageRecord],
    dataset_name: str,
    config: CreRLMeanImageExperimentConfig,
    normalization_stats: NormalizationStats,
) -> float:
    loader = make_dataloader(
        data_root=config.data_root,
        records=records,
        transform=build_transform(
            to_standard_config(config),
            dataset_name=dataset_name,
            train=False,
            normalization_stats=normalization_stats,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )
    total_loss, _ = evaluate_soft_ce(
        model=model,
        loader=loader,
        device=config.device,
        use_amp=config.amp and config.device.startswith("cuda"),
    )
    return total_loss


def evaluate_soft_ce_mean_from_loader(
    *,
    model: nn.Module,
    loader: Any,
    device: str,
    use_amp: bool,
) -> float:
    total_loss, total_items = evaluate_soft_ce(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
    )
    return total_loss / total_items


def to_standard_config(config: CreRLMeanImageExperimentConfig) -> ImageExperimentConfig:
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
        amp=config.amp,
    )


def serialize_config(config: CreRLMeanImageExperimentConfig) -> dict[str, Any]:
    return {
        "method": "crel",
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
        "amp": config.amp,
        "alpha": config.alpha,
        "tobias_strength": config.tobias_strength,
    }


def build_run_name(config: CreRLMeanImageExperimentConfig) -> str:
    mode = "finetune" if not config.freeze_encoder else "head-only"
    pretrained = "pretrained" if config.pretrained else "scratch"
    fold_part = "all-folds" if not config.folds else "-".join(sorted(config.folds))
    normalization_part = (
        ""
        if config.normalization == "imagenet"
        else f"_norm-{sanitize_path_token(config.normalization)}"
    )
    augmentation_part = (
        ""
        if config.augmentation == "basic"
        else f"_aug-{sanitize_path_token(config.augmentation)}"
    )
    readable = (
        f"{config.encoder_name}_crel_{mode}_{pretrained}"
        f"_ens{config.ensemble_size}"
        f"_ep{config.epochs}"
        f"_bs{config.batch_size}"
        f"_seed{config.seed}"
        f"_{fold_part}"
        f"{'_amp' if config.amp else '_fp32'}"
        f"_alpha-{sanitize_path_token(f'{config.alpha:g}')}"
        f"_tobias-{sanitize_path_token(f'{config.tobias_strength:g}')}"
        f"{normalization_part}"
        f"{augmentation_part}"
    )
    fingerprint = hashlib.sha256(
        json.dumps(serialize_config_for_run_name(config), sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return f"{sanitize_path_token(readable)}_{fingerprint}"


def serialize_config_for_run_name(config: CreRLMeanImageExperimentConfig) -> dict[str, Any]:
    payload = serialize_config(config)
    if config.normalization == "imagenet":
        payload.pop("normalization", None)
    if config.augmentation == "basic":
        payload.pop("augmentation", None)
    return payload


def serialize_for_run(
    config: CreRLMeanImageExperimentConfig,
    dataset_names: list[str],
) -> dict[str, Any]:
    return {
        "datasets": dataset_names,
        "config": serialize_config(config),
    }


__all__ = [
    "CreRLMeanImageExperimentConfig",
    "build_run_name",
    "run_dataset_experiment",
    "save_results_summary",
    "serialize_for_run",
    "write_json",
]
