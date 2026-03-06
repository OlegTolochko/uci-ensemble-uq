# UCI Ensemble Uncertainty Quantification

## Image soft-label training

Image datasets should be stored under [data/image](data/image). Compatible datasets can be downloaded here: https://zenodo.org/records/8115942

### Pipeline

- loads annotator votes from each dataset `annotations.json`
- creates per-image target probability distributions from annotator votes
- test data is a held-out fold (each dataset is already split into 5 separate folds), remaining folds are used for training
- trains a classifier (basically any image classifier can be chosen that is supported via torchvision) with soft-target cross entropy
- exports per-image predicted distributions for later uncertainty analysis

### Supported encoders

- `resnet18`
- `resnet50`
- `efficientnet_b0`
- `convnext_tiny`
- `vit_b_16`

The implementation is in [image_pipeline.py](image_pipeline.py) and the CLI runner is [run_image_experiments.py](run_image_experiments.py).

### Default training setup

- pretrained ImageNet encoder
- frozen encoder by default, with a learned classification head
- soft-target cross entropy loss
- fold-based evaluation using the existing `fold1` ... `fold5` directories

### Example commands

Run one dataset with a `resnet18` ensemble:

    python run_image_experiments.py --dataset CIFAR10H --encoder resnet18 --ensemble-size 5

Fine-tune the full encoder instead of only the head:

    python run_image_experiments.py --dataset Benthic --encoder convnext_tiny --ensemble-size 3 --finetune

Quick smoke test on a small subset:

    python run_image_experiments.py --dataset CIFAR10H --encoder resnet18 --ensemble-size 1 --epochs 1 --max-train-samples 64 --max-test-samples 32 --workers 0

### Outputs

Results are written under [out/image](out/image):

- one folder per dataset and encoder
- one subfolder per held-out fold
- `predictions.csv` for each ensemble member
- `ensemble_predictions.csv` for the averaged ensemble probabilities
- `summary.json` with fold metrics and mean cross entropy
- top-level `results.json` summarizing all runs launched by the CLI

Each exported prediction file contains:

- image path
- held-out fold
- target distribution for every class
- predicted distribution for every class
- per-sample cross entropy
- target entropy

## Dataset Overview

| Dataset | Samples | Features | Classes | Label Col | UCI Link | LR Acc | TabPFN Acc | LR Cov | TabPFN Cov |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| Students Dropout | 4,424 | 36 | 3 | -1 | [Link](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) | 0.736 | 0.786 | 0.974 | 0.987 |
| Wine | 178 | 13 | 3 | 0 | [Link](https://archive.ics.uci.edu/dataset/109/wine) | 0.963 | 1.000 | 1.000 | 0.796 |
| Seeds | 210 | 7 | 3 | -1 | [Link](https://archive.ics.uci.edu/dataset/236/seeds) | 0.857 | 0.889 | 0.143 | 0.143 |
| CMC | 1,473 | 9 | 3 | -1 | [Link](https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice) | 0.532 | 0.597 | 0.396 | 0.235 |
| Wall-Robot Navigation | 5,456 | 24 | 4 | -1 | [Link](https://archive.ics.uci.edu/dataset/194/wall+following+robot+navigation+data) | 0.701 | 0.995 | 0.256 | 0.025 |
| Satellite (Statlog) | 6,435 | 36 | 6 | -1 | [Link](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite) | 0.811 | 0.934 | 0.275 | 0.015 |
| Glass Identification | 214 | 9 | 6 | -1 | [Link](https://archive.ics.uci.edu/dataset/42/glass+identification) | 0.631 | 0.892 | 0.169 | 0.000 |
| Image Segmentation | 2,310 | 19 | 7 | -1 | [Link](https://archive.ics.uci.edu/dataset/50/image+segmentation) | 0.945 | 0.983 | 0.078 | 0.000 |
| MFeat-Factors | 2,000 | 216 | 10 | -1 | [Link](https://archive.ics.uci.edu/dataset/72/multiple+features) | 0.975 | 0.975 | 0.885 | 0.000 |
| MFeat-Zernike | 2,000 | 47 | 10 | -1 | [Link](https://archive.ics.uci.edu/dataset/72/multiple+features) | 0.792 | 0.878 | 0.445 | 0.000 |
| Optical Digits | 5,620 | 64 | 10 | -1 | [Link](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) | 0.968 | 0.985 | 0.241 | 0.000 |
| Pen-Based Digits | 10,992 | 16 | 10 | -1 | [Link](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits) | 0.940 | 0.996 | 0.106 | 0.000 |

Note: For all experiments the following hyperparameters were used:
```python
arguments = {
    ...
    "test_size": 0.3,
    "calibration_size": 0.0,
    "ensemble_size": 25,
    "random_state": 42,
    "sampling_mode": "shared",
}
```
