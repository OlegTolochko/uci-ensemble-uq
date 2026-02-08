import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from probly.metrics import coverage_convex_hull


def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


def predict(model, X):
    return model.predict_proba(X)


def entropy_histogram(probs, bins=10, save_path="out/entropy_histogram.png"):
    ent = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    plt.hist(ent, bins=bins)
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.title("Prediction Entropy Distribution")
    plt.show()
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


def pipeline(
    data_path="data/iris/iris.data",
    label_column=-1,
    test_size=0.3,
    ensemble_size=10,
    random_state=42,
):
    X, y = load_data(data_path=data_path, label_column=label_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = train_model(X_train, y_train)
    probs_test = predict(model, X_test)
    entropy_histogram(probs_test)

    probs_train = predict(model, X_train)
    sampled_labels = sample_labels(
        probs_train, ensemble_size=ensemble_size, random_state=random_state
    )
    ensemble = train_ensemble(X_train, sampled_labels)

    classes = np.unique(y)
    n_classes = len(classes)

    check_convex_hull_coverage(ensemble, X_test, probs_test, n_classes=n_classes)


def train_ensemble(X, y_ensemble):
    ensemble = []
    for ensemble_labels in y_ensemble.T:
        model = MLPClassifier(max_iter=1000, random_state=np.random.randint(10000))
        model.fit(X, ensemble_labels)
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
        "data_path": "data/iris/iris.data",
        "label_column": 0,
        "test_size": 0.3,
        "ensemble_size": 10,
        "random_state": 42,
    }
    pipeline(**arguments)
