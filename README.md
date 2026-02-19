# Dataset Overview

| Dataset | Samples | Features | Classes | Label Col | UCI Link | LR Acc | TabPFN Acc | Convex Hull Coverage |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Students Dropout and Academic Success | 4,424 | 36 | 3 | -1 | [Link](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) | - | - | - |
| Wine | 178 | 13 | 3 | 0 | [Link](https://archive.ics.uci.edu/dataset/109/wine) | - | - | - |
| Wine Quality | 4,898 | 11 | 6-7 | -1 | [Link](https://archive.ics.uci.edu/dataset/186/wine+quality) | - | - | - |
| Yeast | 1,484 | 8 | 10 | -1 | [Link](https://archive.ics.uci.edu/dataset/110/yeast) | - | - | - |
| Letter Recognition | 20,000 | 16 | 26 | 0 | [Link](https://archive.ics.uci.edu/dataset/59/letter+recognition) | - | - | - |
| Abalone | 4,177 | 8 | 28 | -1 | [Link](https://archive.ics.uci.edu/dataset/1/abalone) | - | - | - |

Note: For all experiments following hyperparameters were used:
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