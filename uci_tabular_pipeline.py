import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier
from probly.metrics import coverage_convex_hull
from sklearn.metrics import accuracy_score
from tabpfn_extensions.many_class import ManyClassClassifier

warnings.filterwarnings("ignore")


def _fit_logistic(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def train_model(X, y, model_type="logistic"):
    if model_type == "tabpfn":
        base = TabPFNClassifier(ignore_pretraining_limits=True)
        model = ManyClassClassifier(estimator=base, alphabet_size=10)
        model.fit(X, y)
    else:
        model = _fit_logistic(X, y)
    return model


def predict(model, X):
    return model.predict_proba(X)


def entropy_histogram(
    probs, bins=10, save_path="out/entropy_histogram.png", dataset_title="Dataset"
):
    K = probs.shape[1]
    ent = -np.sum(probs * np.log(probs + 1e-12) / np.log(K), axis=1)
    plt.figure()
    plt.hist(ent, bins=bins)
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.title(f"Prediction Entropy Distribution ({dataset_title})")
    plt.savefig(save_path)
    plt.close()


def load_data(
    data_path="data/iris/iris.data",
    label_column=-1,
    sep=None,
    header=None,
    usecols=None,
    id_column=None,
    encode_first_column=False,
):
    df = pd.read_csv(data_path, sep=sep, header=header)

    if id_column is not None:
        df = df.drop(columns=df.columns[id_column])

    if encode_first_column:
        first_col = df.columns[0]
        if df[first_col].dtype == object or str(df[first_col].dtype) == "str":
            le = LabelEncoder()
            df[first_col] = le.fit_transform(df[first_col])

    if usecols is not None:
        df = df[usecols]

    y = df.iloc[:, label_column].values
    X = df.drop(columns=df.columns[label_column]).values.astype(float)
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


def can_stratify(y, test_size):
    _, counts = np.unique(y, return_counts=True)
    min_samples_needed = 2 if test_size < 1 else 1
    return all(c >= min_samples_needed for c in counts)


def pipeline(
    data_path="data/iris/iris.data",
    label_column=-1,
    base_model="logistic",
    test_size=0.3,
    calibration_size=0.25,
    ensemble_size=10,
    sampling_mode="per_model",
    random_state=42,
    sep=None,
    header=None,
    id_column=None,
    encode_first_column=False,
):
    X, y = load_data(
        data_path=data_path,
        label_column=label_column,
        sep=sep,
        header=header,
        id_column=id_column,
        encode_first_column=encode_first_column,
    )
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    use_stratify = can_stratify(y, test_size)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if use_stratify else None,
    )

    if calibration_size > 0:
        use_stratify_calib = can_stratify(y_train_full, calibration_size)
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train_full,
            y_train_full,
            test_size=calibration_size,
            random_state=random_state,
            stratify=y_train_full if use_stratify_calib else None,
        )
    else:
        X_train, y_train = X_train_full, y_train_full

    model = train_model(X_train, y_train, model_type=base_model)

    if calibration_size > 0 and base_model != "tabpfn":
        calibrated = fit_isotonic_calibrator(model, X_calib, y_calib)
        probs_test = predict(calibrated, X_test)
        probs_train = predict(calibrated, X_train)
    else:
        probs_test = predict(model, X_test)
        probs_train = predict(model, X_train)

    dataset_name = data_path.split("/")[-2]
    entropy_histogram(
        probs_test,
        save_path=f"out/entropy_histogram_{dataset_name}_{base_model}.png",
        dataset_title=f"{dataset_name} ({base_model})",
    )

    y_pred_test = np.argmax(probs_test, axis=1)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Base model accuracy: {accuracy:.3f}")

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

    convex_hull_cov = check_convex_hull_coverage(
        ensemble, X_test, probs_test, n_classes=n_classes
    )
    return accuracy, convex_hull_cov


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
        "base_model": "tabpfn",  # "logistic" or "tabpfn"
        "data_path": "data/wine/wine.data",
        "label_column": 0,  # -1 for iris and pendigits, 0 for wine, abalone and letter-recognition
        "test_size": 0.3,
        "calibration_size": 0.0,
        "ensemble_size": 25,
        "random_state": 42,
        "sampling_mode": "shared",  # "per_model" or "shared"
    }
    pipeline(**arguments)
