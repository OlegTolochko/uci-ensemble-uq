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

### Entmax Image Pipeline

- Main runner: [run_image_entmax_simple.py](run_image_entmax_simple.py)
- Main pipeline: [image_entmax_pipeline_simple.py](image_entmax_pipeline_simple.py)
- Results helper: [image_results.py](image_results.py)

### Supported args

| Argument            |                                            Default value | Description                                                                                                    |
| ------------------- | -------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------- |
| `--dataset`         |                                                   `None` | if not set, all datasets found in `data_root` are used, can be passed multiple times                           |
| `--encoder`         |                                               `resnet18` | which torchvision encoder to use (e.g. also resnet50, check DEFAULT_ENCODERS in main pipeline for all options) |
| `--ensemble-size`   |                                                     `25` | number of independently initialized models trained on the same split                                           |
| `--epochs`          |                                                     `20` | number of training epochs                                                                                      |
| `--batch-size`      |                                                     `32` | batch size used for train, validation and test dataloaders                                                     |
| `--lr`              |                                                   `1e-3` | learning rate for AdamW                                                                                        |
| `--weight-decay`    |                                                   `1e-4` | weight decay for AdamW                                                                                         |
| `--validation-size` |                                                    `0.1` | fraction of the non-test data used as validation set                                                           |
| `--patience`        |                                                      `4` | number of epochs with no improvement on val set after which training is stopped                                |
| `--seed`            |                                                     `42` | random seed used for splitting and model initialization                                                        |
| `--workers`         |                                                      `4` | number of dataloader worker processes                                                                          |
| `--device`          | `cuda` if available, else `mps` if available, else `cpu` | which device to run training and inference on                                                                  |
| `--finetune`        |                                                  `False` | if not set, encoder is frozen and only classification head is trained                                          |
| `--no-pretrained`   |                                                  `False` | if set, encoder is initialized with random weights instead of imagenet pretrained weights                      |
| `--test-fold`       |                                                  `fold1` | which fold to use as test set, options are: `fold1`, `fold2`, `fold3`, `fold4`, `fold5`                        |
| `--output-root`     |                                              `out/image` | root directory where run outputs are written                                                                   |
| `--data-root`       |                                             `data/image` | root directory containing the image datasets                                                                   |
| `--augmentation`    |                                                  `basic` | basic is currently just `RandomHorizontalFlip`                                                                 |
| `--dropout`         |                                                    `0.0` | dropout rate for classification head, applied after global average pooling and before linear layer             |
| `--entmax-alpha`    |                                                    `1.5` | alpha parameter for entmax, controls sparsity output logits, alpha=1 is softmax, alpha=2 is sparsemax          |

### Full Pipeline

run_image_entmax_simple:
1. Initialize Argument Parser
2. Construct config from passed arguments
3. Create run directory and write config to .json
4. For each dataset run: run_dataset_experiment(dataset_name, config)
5. Save run result summary for each dataset


image_entmax_pipeline_simple, run_dataset_experiment:
1. load class_names + dataset records (load_image_dataset)
	- Each dataset has an annotations.json, which links an image path to its vote counts, structure: {"record_n": {"annotations": {"image_path": ..., "class_label": ..., "created_at": ...}, ...}, ...}
	- For each annotation get image path + class label, count number of times a class was voted for a given image path
	- class_counts is a dict (associated with an image_path) that contains a mapping from class_names with their respective counts e.g. {"car": 1, "house": 7, "tree": 2}
	- transform counts to probabilities, wrap each Image into a ImageRecord (stores image path, fold it was in and target distribution)
2. Put all records that match chosen test fold in the test set, all others in the train set
3. Split train set into train and validation sets (additional shuffling of records given the seed, default split is 90/10)
4. For each ensemble member index:
	1. Construct own directory inside run directory
	2. train member given set config:
		1. Train single model:
			1. Create train and validation dataloaders with a transform recipe built in build_transform (Image resize to encoder input and RandomHorizontalFlip)
			2. Initialize model:
				1. If pretrained model is utilized, imagenet weights are loaded for the specified model
				2. replace old classification head with identity and get number of in features to classification head (replace_classification_head_with_identity)
				3. Bulid new classification head, single Linear Layer with num_classes neurons and in_features weights + 1 weights per neuron (with optional dropout before linear layer)
				4. if only classification head is supposed to be trained (so no --finetune flag was passed) encoder parameters are frozen
			3. Initialize AdamW optimizer and Loss function (Fenchel-Young loss formulation, required since entmax uses different mapping from logits to probabilities)
			4. for specified number of epochs:
				1. Standard torch model training setup, update model weights
				2. Then also run inference once for validation data
				3. check if validation loss is smaller than previous best val loss, if so start counting up to early stopping threshold
		2. run trained model on test data and compute predicted probability distributions with entmax
		3. compute mean cross entropy loss for this member
		4. save model, all predictions in .csv (image path, fold, cross_entropy, target_entropy + target and predicted probabilities for each class) and a small summary
	3. compute mean ensemble-wise probability predictions and compute ensemble cross entropy
	4. save results for fold + dataset, as well as a summary and the utilized config

### Supported encoders

- `resnet18`
- `resnet50`
- `efficientnet_b0`
- `convnext_tiny`
- `vit_b_16`

### Example commands

Run one dataset with a `resnet18` ensemble:

    python run_image_entmax_simple.py --dataset CIFAR10H --encoder resnet18 --ensemble-size 5

Fine-tune the full encoder instead of only the head:

    python run_image_entmax_simple.py --dataset Benthic --encoder convnext_tiny --ensemble-size 3 --finetune

Run a single held-out fold:

    python run_image_entmax_simple.py --dataset CIFAR10H --encoder resnet18 --ensemble-size 5 --test-fold fold1

### Outputs

Results are written under [out/image](out/image):

- one run directory per executed configuration
- one folder per dataset and encoder inside the run directory
- one subfolder for the chosen held-out fold
- `predictions.csv` for each ensemble member
- `ensemble_predictions.csv` for the averaged ensemble probabilities
- `summary.json` with fold metrics and mean cross entropy
- top-level `results.json` summarizing all datasets in the run

Each exported prediction file contains:

- image path
- held-out fold
- target distribution for every class
- predicted distribution for every class
- per-sample cross entropy
- target entropy

`image_results.py` reads one run directory and provides:

- `iter_dataset_rows(run_dir)` to iterate row by row over dataset results
- `get_latex_table(run_dir)` to build a LaTeX table

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
