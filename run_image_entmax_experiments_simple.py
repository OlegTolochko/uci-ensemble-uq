import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any
import torch

from image_entmax_pipeline import (
    EntmaxImageExperimentConfig,
    build_run_name,
    run_dataset_experiment,
    save_results_summary,
    serialize_for_run,
)

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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--encoder", default="resnet18", choices=sorted(DEFAULT_ENCODERS.keys()))
    parser.add_argument("--ensemble-size", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4) # number of epochs with no improvement on val set after which training is stopped
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--finetune", action="store_true") # if not set, encoder is frozen and only classification head is trained
    parser.add_argument("--no-pretrained", action="store_true") # if set, encoder is initialized with random weights instead of imagenet pretrained weights
    parser.add_argument("--test-fold", default="fold1") # which fold to use as test set, options are: fold1, fold2, fold3, fold4, fold5
    parser.add_argument("--output-root", type=Path, default=Path("out/image"))
    parser.add_argument("--data-root", type=Path, default=Path("data/image"))
    parser.add_argument("--augmentation", choices=["none", "basic"], default="basic") # basic is currently just RandomHorizontalFlip
    parser.add_argument("--dropout", type=float, default=0.0) # dropout rate for classification head, applied after global average pooling and before linear layer
    parser.add_argument("--entmax-alpha", type=float, default=1.5) # alpha parameter for entmax, controls sparsity output logits, alpha=1 is softmax, alpha=2 is sparsemax
    return parser


def load_all_image_datasets(data_root: Path) -> list[str]:
    return sorted(path.name for path in data_root.iterdir() if path.is_dir())

 
def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = build_parser().parse_args()
    datasets = args.datasets or load_all_image_datasets(args.data_root)
    config = EntmaxImageExperimentConfig(
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
        augmentation=args.augmentation,
        classifier_dropout=args.dropout,
        entmax_alpha=args.entmax_alpha,
    )

    run_name = build_run_name(config)
    run_root = config.output_root / run_name
    config = replace(config, output_root=run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "config.json", serialize_for_run(config, datasets))

    print(f"Run directory: {run_root}")
    results = []
    for dataset_name in datasets:
        print(f"\nRunning entmax on {dataset_name}")
        result = run_dataset_experiment(dataset_name, config)
        print(f"Mean ensemble cross entropy: {result.mean_ensemble_cross_entropy:.4f}")
        results.append(result)
        save_results_summary(run_root, results)

    print("\nEnsemble cross entropy results:")
    for result in results:
        print(f"{result.dataset_name}: ensemble CE={result.mean_ensemble_cross_entropy:.4f}")


if __name__ == "__main__":
    main()
