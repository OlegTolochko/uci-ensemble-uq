# Ensemble Uncertainty Quantification

## Image dataset overview

| Dataset       | size  | classes | avg. entropy | type                                                                        | input size | Mean Cross entropy |
| ------------- | ----- | ------- | ------------ | --------------------------------------------------------------------------- | ---------- | --------------------------- |
| benthic       | 4867  | 8       | 0.340        | image (images from the seafloor and consists of underwater flora and fauna) | 112x112    | 1.2707                      |
| cifar-10h     | 10000 | 10      | 0.154        | image                                                                       | 32x32      | 0.7193                      |
| MiceBone      | 7240  | 4       | 0.319        | image (Second-<br>Harmonic-Generation images of collagen fibers)            | 224x224    | 0.6009                      |
| pig           | 10237 | 4       | 0.735        | image (tail images form european farms)                                     | 96x96      | 1.2544                      |
| plankton      | 12280 | 10      | 0.163        | image (underwater plankton images)                                          | 96x96      | 0.6446                      |
| qualityMRI    | 310   | 2       | 0.556        | image (MRI images)                                                          | 224x224    | 0.6441                      |
| synthetic     | 15000 | 6       | 0.584        | image (images that contain 1 colored circle on a black background)          | 224x224    | 0.7084                      |
| TreeVersity#1 | 9489  | 6       | 0.266        | image (plant images, single label per image)                                | 224x224    | 0.7351                      |
| TreeVersity#6 | 9826  | 6       | 0.742        | image (plant images, possibly multiple labels per image)                    | 224x224    | 1.1153                      |
| turkey        | 8040  | 3       | 0.196        | image (images of turkeys and their injuries)                                | 192x192    | 0.5712                      |

(Config for Mean Cross Entropy: resnet18, 5 Fold CV mean, frozen encoder weights, 10 epochs)

Image datasets should be stored under [data/image](data/image). Used datasets can be downloaded here: https://zenodo.org/records/8115942

### Pipeline Overview

- loads annotator votes from each dataset `annotations.json`
- creates per-image target probability distributions from annotator votes
- test data is a held-out fold (each dataset is already split into 5 separate folds), remaining folds are used for training
- trains a classifier (basically any image classifier can be chosen that is supported via torchvision) with soft-target cross entropy
- exports per-image predicted distributions for later uncertainty analysis


### Detailed Pipeline Overview
<details>
<summary>Expand for Detailed Pipeline</summary>

- Initialize Config for running experimenets (run_image_experiments.py)
- For each dataset given (default is all datasets) run experiment pipeline (run_dataset_experiment), default trains 5 models, one per test fold, --fold fold1 would only use the first fold for testing
- load class_names + dataset records (load_image_dataset)
	- Each dataset has an annotations.json, which links an image path to its vote counts, structure: {"record_n": {"annotations": {"image_path": ..., "class_label": ..., "created_at": ...}, ...}, ...}
	- For each annotation get image path + class label, count number of times a class was voted for a given image path
	- transform counts to probabilities, wrap each Image into a ImageRecord (stores image path, fold it was in and target distribution)
- For each given fold put all records that much this specific fold in the test set, all others in the train set
- Split train set into train and validation sets
- For the set number of ensemble members, train a model (each with different seed)
	1. create train and validation dataloader, with set batch_size and transformations (currently normalization with imagenet mean + std, since pretrained models were trained on imagenet, as well as resize to expected input image size). Additionally also RandomHorizontalFlip currently for the train set
	2. Initialize Model, based on passed model name (e.g. resnet18, resnet50, vit_b_16, etc.), replace classification head with linear layer and optionally freeze all layers leading up to classification head (--finetune in config if weights should not be frozen)
	3. Set up optimizer (adamW) + loss (soft target cross entropy)
	4. Begin training loop for set amount of epochs, (run_epoch for both training set, where gradients are calculated and weights are updated; once for validation set, where no weights are updated)
	5. If validation loss is higher than the best validation loss recorded yet for a set amount of epochs, stop training early and load best model state and save model
- Test model performance (calculate mean cross entropy) on held out test old and save individual prediction results to a csv table
- Save result summary for each dataset (summary.json) across all fold results
</details>

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

## UCI Dataset Overview

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
