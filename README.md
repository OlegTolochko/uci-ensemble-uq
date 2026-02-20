# UCI Ensemble Uncertainty Quantification

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

## Notes on Convex Hull Coverage with Shared Sampling

The `shared` sampling mode trains all ensemble members on the **same** set of sampled labels, with
diversity coming only from different random MLP weight initialisations. This limits the spread of
the ensemble's predicted probability distributions, which in turn limits how large the convex hull
can be in the probability simplex.

Key observations from the dataset search:

- **3 classes** – Coverage is generally healthy (0.14–1.00) because the 2-dimensional simplex is
  easy to cover even with modest ensemble diversity.
- **6 classes** – Coverage starts dropping; Satellite (0.275) and Glass (0.169) still work for LR
  thanks to moderate base-model difficulty and reasonable feature counts.
- **10 classes** – MFeat-Factors stands out at 0.885 LR coverage. Its 216 features give the MLP
  enough degrees of freedom to find meaningfully different solutions from different random
  initialisations, even with identical training labels.
- **>10 classes** – No dataset tested achieved non-zero convex hull coverage under shared sampling.
  The probability simplex becomes too high-dimensional for the tight ensemble to cover.

### TabPFN vs Logistic Regression Coverage

TabPFN consistently achieves higher accuracy than logistic regression, but this **hurts** convex
hull coverage under shared sampling. Because TabPFN produces more confident (peaked) probability
distributions, the target vector is harder for the MLP ensemble's convex hull to contain. This
effect is especially pronounced for 6+ classes, where TabPFN coverage drops to 0.000 on every
dataset despite strong accuracy. For 3-class datasets the coverage remains non-zero but is
generally lower than the LR counterpart (e.g. CMC: 0.396 LR vs 0.235 TabPFN).

### Factors that promote non-zero coverage

1. **Balanced classes** – rare classes with very few samples create probability dimensions the
   ensemble consistently mis-covers.
2. **Many features relative to classes** – more features → more room for the MLP to find diverse
   solutions from different random initialisations.
3. **Moderate base-model accuracy** – overly confident models produce very peaked target
   distributions that are hard for the ensemble to bracket.