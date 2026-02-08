import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier



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

def load_data(data_path="data/iris/iris.data"):
    df = pd.read_csv(data_path, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def sample_labels(probs, ensemble_size=10, random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples, n_classes = probs.shape
    labels = np.array([
        rng.choice(n_classes, size=ensemble_size, p=p) for p in probs
    ])
    return labels

def pipeline(test_size=0.3, random_state=42):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = train_model(X_train, y_train)
    probs = predict(model, X_test)
    entropy_histogram(probs)

    sampled_labels = sample_labels(probs)


def main():
    pipeline()


if __name__ == "__main__":
    main()
