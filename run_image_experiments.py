from __future__ import annotations

import argparse
from pathlib import Path

import torch
from image_augmentations import AUGMENTATION_MODES
from image_loss_variants import PROB_REGULARIZERS

from image_pipeline import (
    ImageExperimentConfig,
    build_run_name,
    discover_image_datasets,
    list_available_encoders,
    run_dataset_experiment,
    save_results_summary,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train soft-label image classification ensembles on datasets in data/image/."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset name under data/image/. Repeat to run multiple datasets. Defaults to all datasets.",
    )
    parser.add_argument(
        "--encoder",
        default="resnet18",
        choices=list_available_encoders(),
        help="Backbone encoder used for the classifier head.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Number of independently trained models per held-out test fold.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--prob-regularizer",
        choices=PROB_REGULARIZERS,
        default="none",
        help="Optional probability-space penalty added to the soft-target CE.",
    )
    parser.add_argument(
        "--prob-regularizer-weight",
        type=float,
        default=0.0,
        help="Weight for the probability-space penalty.",
    )
    parser.add_argument(
        "--entropy-bonus-weight",
        type=float,
        default=0.0,
        help="Reward predictive entropy to discourage overconfident outputs.",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=0.0,
        help="Strength of the DARE regularizer. Disabled at 0.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Optional dropout applied in the classifier head.",
    )
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Explicit device, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Update encoder weights instead of training only the classification head.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet initialization for the encoder.",
    )
    parser.add_argument(
        "--fold",
        action="append",
        dest="folds",
        help="Held-out test fold to evaluate, for example fold1. Repeat to run multiple folds.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("out/image"),
        help="Directory used for metrics, summaries, and prediction exports.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/image"),
        help="Directory containing the image datasets.",
    )
    parser.add_argument(
        "--normalization",
        default="imagenet",
        choices=["imagenet", "dataset"],
        help=(
            "Normalization mode. "
            "'imagenet' keeps torchvision ImageNet stats. "
            "'dataset' caches 5 RGB mean/std sets per dataset, one for each held-out fold, "
            "computed from all non-held-out folds."
        ),
    )
    parser.add_argument(
        "--augmentation",
        default="basic",
        choices=AUGMENTATION_MODES,
        help=(
            "Training-time augmentation policy. "
            "'basic' keeps the current random horizontal flip baseline. "
            "'dcic_auto' mirrors the dcic paper setup by disabling augmentation for "
            "CIFAR10H and QualityMRI and using the fuller bundle elsewhere."
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mixed precision on CUDA for faster training.",
    )
    return parser


def serialize_for_run(config: ImageExperimentConfig, dataset_names: list[str]) -> dict:
    return {
        "datasets": dataset_names,
        "config": {
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
            "prob_regularizer": config.prob_regularizer,
            "prob_regularizer_weight": config.prob_regularizer_weight,
            "entropy_bonus_weight": config.entropy_bonus_weight,
            "amp": config.amp,
        },
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    dataset_names = args.datasets or discover_image_datasets(args.data_root) 
    base_config = ImageExperimentConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        encoder_name=args.encoder,
        pretrained=not args.no_pretrained,
        freeze_encoder=not args.finetune,
        ensemble_size=args.ensemble_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        prob_regularizer=args.prob_regularizer,
        prob_regularizer_weight=args.prob_regularizer_weight,
        entropy_bonus_weight=args.entropy_bonus_weight,
        lambda_reg=args.lambda_reg,
        classifier_dropout=args.dropout,
        validation_size=args.validation_size,
        early_stopping_patience=args.patience,
        num_workers=args.workers,
        device=args.device,
        seed=args.seed,
        folds=args.folds,
        normalization=args.normalization,
        augmentation=args.augmentation,
        amp=args.amp,
    )
    run_name = build_run_name(base_config)
    run_output_root = args.output_root / run_name
    config = ImageExperimentConfig(
        data_root=base_config.data_root,
        output_root=run_output_root,
        encoder_name=base_config.encoder_name,
        pretrained=base_config.pretrained,
        freeze_encoder=base_config.freeze_encoder,
        ensemble_size=base_config.ensemble_size,
        batch_size=base_config.batch_size,
        epochs=base_config.epochs,
        learning_rate=base_config.learning_rate,
        weight_decay=base_config.weight_decay,
        prob_regularizer=base_config.prob_regularizer,
        prob_regularizer_weight=base_config.prob_regularizer_weight,
        entropy_bonus_weight=base_config.entropy_bonus_weight,
        lambda_reg=base_config.lambda_reg,
        classifier_dropout=base_config.classifier_dropout,
        validation_size=base_config.validation_size,
        early_stopping_patience=base_config.early_stopping_patience,
        num_workers=base_config.num_workers,
        device=base_config.device,
        seed=base_config.seed,
        folds=base_config.folds,
        normalization=base_config.normalization,
        augmentation=base_config.augmentation,
        amp=base_config.amp,
    )
    write_json(run_output_root / "config.json", serialize_for_run(config, dataset_names))

    print(f"Run directory: {run_output_root}")

    results = []
    for dataset_name in dataset_names:
        print(f"\nRunning {config.encoder_name} on {dataset_name}")
        result = run_dataset_experiment(dataset_name=dataset_name, config=config)
        print(
            f"  mean ensemble cross entropy: {result.mean_ensemble_cross_entropy:.4f}"
        )
        results.append(result)
        save_results_summary(run_output_root, results)

    print("\nSummary")
    print("=" * 60)
    for result in results:
        print(
            f"{result.dataset_name}: ensemble CE="
            f"{result.mean_ensemble_cross_entropy:.4f}"
        )


if __name__ == "__main__":
    main()
