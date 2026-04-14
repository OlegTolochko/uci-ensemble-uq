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

@dataclass(frozen=True, slots=True)
class EntmaxImageExperimentConfig:
    data_root: Path
    output_root: Path
    encoder_name: str
    pretrained: bool
    freeze_encoder: bool
    ensemble_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_size: float
    early_stopping_patience: int
    num_workers: int
    device: str
    seed: int
    test_fold: str
    augmentation: str
    classifier_dropout: float
    entmax_alpha: float

def config_to_dict(config: EntmaxImageExperimentConfig) -> dict[str, Any]:
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
        "test_fold": config.test_fold,
        "augmentation": config.augmentation,
        "classifier_dropout": config.classifier_dropout,
        "entmax_alpha": config.entmax_alpha,
    }


def build_run_name(config: EntmaxImageExperimentConfig) -> str:
    mode = "finetune" if not config.freeze_encoder else "linear"
    pretrained = "pretrained" if config.pretrained else "scratch"
    fold_part = sanitize_path_token("-".join(config.folds or ["all-folds"]))
    augmentation_part = (
        "" if config.augmentation == "basic" else f"_aug-{config.augmentation}"
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


def run_dataset_experiment(dataset_name: str, config: EntmaxImageExperimentConfig) -> DatasetResult:
    output_dir = config.output_root / dataset_name / config.encoder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    write_json(output_dir / "config.json", serialize_config(config))

    if summary_path.exists():
        return load_dataset_result(summary_path)

    class_names, records = load_image_dataset(config.data_root / dataset_name)

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