import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.preprocessing import LabelEncoder
from probly.metrics import coverage_convex_hull


def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


def predict(model, X):
    return model.predict_proba(X)


def entropy_histogram(probs, bins=10, save_path="out/entropy_histogram.png", dataset_title="Dataset"):
    K = probs.shape[1]
    ent = -np.sum(probs * np.log(probs + 1e-12) / np.log(K), axis=1)
    plt.hist(ent, bins=bins)
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.title(f"Prediction Entropy Distribution ({dataset_title})")
    plt.savefig(save_path)


def load_data(data_path="data/iris/iris.data", label_column=-1):
    df = pd.read_csv(data_path, header=None)
    y = df.iloc[:, label_column].values
    X = df.drop(columns=df.columns[label_column]).values
    return X, y


def sample_labels(probs, ensemble_size=10, random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples, n_classes = probs.shape
    labels = np.array([rng.choice(n_classes, size=ensemble_size, p=p) for p in probs])
    return labels


def sample_labels_single(probs, random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples, n_classes = probs.shape
    labels = np.array([rng.choice(n_classes, p=p) for p in probs])
    return labels


def fit_isotonic_calibrator(base_model, X_calib, y_calib):
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="isotonic",
    )
    calibrator.fit(X_calib, y_calib)
    return calibrator

    

def pipeline(
    data_path="data/iris/iris.data",
    label_column=-1,
    test_size=0.3,
    calibration_size=0.25,
    ensemble_size=10,
    sampling_mode="per_model",
    random_state=42,
):
    X, y = load_data(data_path=data_path, label_column=label_column)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if calibration_size > 0:
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train_full,
            y_train_full,
            test_size=calibration_size,
            random_state=random_state,
            stratify=y_train_full,
        )
    else:
        X_train, y_train = X_train_full, y_train_full
    
    base_model = train_model(X_train, y_train)

    if calibration_size > 0:
        calibrated_model = fit_isotonic_calibrator(base_model, X_calib, y_calib)
        probs_test = predict(calibrated_model, X_test)
        probs_train = predict(calibrated_model, X_train)
    else:
        probs_test = predict(base_model, X_test)
        probs_train = predict(base_model, X_train)

    entropy_histogram(
        probs_test,
        save_path=f"out/entropy_histogram_{data_path.split('/')[-2]}.png",
        dataset_title=data_path.split('/')[-2],
    )

    if sampling_mode == "per_model":
        sampled_labels = sample_labels(
            probs_train, ensemble_size=ensemble_size, random_state=random_state
        )
        ensemble = train_ensemble(X_train, sampled_labels, random_state=random_state)
    elif sampling_mode == "shared":
        shared_labels = sample_labels_single(probs_train, random_state=random_state)
        ensemble = train_ensemble_shared(
            X_train,
            shared_labels,
            ensemble_size=ensemble_size,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

    classes = np.unique(y)
    n_classes = len(classes)

    check_convex_hull_coverage(ensemble, X_test, probs_test, n_classes=n_classes)


def train_ensemble(X, y_ensemble, random_state=42):
    rng = np.random.default_rng(random_state)
    ensemble = []
    for ensemble_labels in y_ensemble.T:
        model = MLPClassifier(max_iter=1000, random_state=int(rng.integers(10000)))
        model.fit(X, ensemble_labels)
        ensemble.append(model)
    return ensemble


def train_ensemble_shared(X, y, ensemble_size=10, random_state=42):
    rng = np.random.default_rng(random_state)
    ensemble = []
    for _ in range(ensemble_size):
        model = MLPClassifier(max_iter=1000, random_state=int(rng.integers(10000)))
        model.fit(X, y)
        ensemble.append(model)
    return ensemble


def ensemble_predict_proba(ensemble, X, n_classes):
    all_probs = []
    for model in ensemble:
        p_raw = model.predict_proba(X)
        p = np.zeros((X.shape[0], n_classes))
        for i, c in enumerate(model.classes_):
            p[:, int(c)] = p_raw[:, i]
        all_probs.append(p)
    return np.stack(all_probs, axis=1)


def check_convex_hull_coverage(ensemble, X_test, targets, n_classes):
    probs = ensemble_predict_proba(ensemble, X_test, n_classes)
    cov = coverage_convex_hull(probs, targets)
    print(f"Convex hull coverage: {cov:.3f}")
    return cov


if __name__ == "__main__":
    arguments = {
        "data_path": "data/wine/wine.data",
        "label_column": 0,  # -1 for iris, 0 for wine and abalone
        "test_size": 0.3,
        "calibration_size": 0.0,
        "ensemble_size": 25,
        "random_state": 43,
        "sampling_mode": "shared",  # "per_model" or "shared"
    }
    pipeline(**arguments)
