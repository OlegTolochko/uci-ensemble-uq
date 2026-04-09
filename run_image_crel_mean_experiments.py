from __future__ import annotations

import argparse
from pathlib import Path

import torch

from image_augmentations import AUGMENTATION_MODES
from image_crel_mean_pipeline import (
    CreRLMeanImageExperimentConfig,
    build_run_name,
    run_dataset_experiment,
    save_results_summary,
    serialize_for_run,
    write_json,
)
from image_pipeline import discover_image_datasets, list_available_encoders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CreRL image ensembles with mean train CE / mean relative-likelihood early stopping."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset name under data/image/. Defaults to all datasets.",
    )
    parser.add_argument(
        "--encoder",
        default="resnet18",
        choices=list_available_encoders(),
        help="Backbone encoder used for the classifier head.",
    )
    parser.add_argument("--ensemble-size", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else  "mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--tobias-strength", type=float, default=100.0)
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
        help="Held-out test fold for evaluation, for example fold1.",
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
    )
    parser.add_argument(
        "--augmentation",
        default="basic",
        choices=AUGMENTATION_MODES,
        help="Training-time augmentation policy for both h_ml and all ensemble members.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout applied in the classifier head.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mixed precision on CUDA for faster training.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_names = args.datasets or discover_image_datasets(args.data_root)
    base_config = CreRLMeanImageExperimentConfig(
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
        validation_size=args.validation_size,
        early_stopping_patience=args.patience,
        num_workers=args.workers,
        device=args.device,
        seed=args.seed,
        folds=args.folds,
        normalization=args.normalization,
        augmentation=args.augmentation,
        classifier_dropout=args.dropout,
        amp=args.amp,
        alpha=args.alpha,
        tobias_strength=args.tobias_strength,
    )
    run_name = build_run_name(base_config)
    run_output_root = args.output_root / run_name
    config = CreRLMeanImageExperimentConfig(
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
        validation_size=base_config.validation_size,
        early_stopping_patience=base_config.early_stopping_patience,
        num_workers=base_config.num_workers,
        device=base_config.device,
        seed=base_config.seed,
        folds=base_config.folds,
        normalization=base_config.normalization,
        augmentation=base_config.augmentation,
        classifier_dropout=base_config.classifier_dropout,
        amp=base_config.amp,
        alpha=base_config.alpha,
        tobias_strength=base_config.tobias_strength,
    )
    write_json(run_output_root / "config.json", serialize_for_run(config, dataset_names))

    print(f"Run directory: {run_output_root}")

    results = []
    for dataset_name in dataset_names:
        print(f"\nRunning {config.encoder_name} CreRL on {dataset_name}")
        result = run_dataset_experiment(dataset_name=dataset_name, config=config)
        print(f"  mean ensemble cross entropy: {result.mean_ensemble_cross_entropy:.4f}")
        results.append(result)
        save_results_summary(run_output_root, results)

    print("\nSummary")
    print("=" * 60)
    for result in results:
        print(f"{result.dataset_name}: ensemble CE={result.mean_ensemble_cross_entropy:.4f}")


if __name__ == "__main__":
    main()
