from model import LogisticRegressionParallel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression


def load_data(X_path: str, y_path: str, n_samples: int = -1, n_features: int = -1):
    """
    Load and optionally subsample data from .npy files.

    Args:
        X_path (str): Path to features (.npy file).
        y_path (str): Path to labels (.npy file).
        n_samples (int): Number of samples to use (-1 = use all).
        n_features (int): Number of features to use (-1 = use all).

    Returns:
        tuple[list[list[float]], list[int]]: (X, y)
    """
    X_raw = np.load(X_path)
    y_raw = np.load(y_path)

    n_samples = X_raw.shape[0] if n_samples < 0 else min(X_raw.shape[0], n_samples)
    n_features = X_raw.shape[1] if n_features < 0 else min(X_raw.shape[1], n_features)

    # Random subsampling
    if n_samples < X_raw.shape[0]:
        idx = np.random.permutation(X_raw.shape[0])
        X_raw = X_raw[idx]
        y_raw = y_raw[idx]
        X_raw = X_raw[:n_samples]
        y_raw = y_raw[:n_samples]

    X = X_raw[:, :n_features].astype(float).tolist()
    y = y_raw.astype(int).tolist()

    return X, y


# =============================== CONFIG ===============================
X_PATH = "../data/1m_100/X_1m_100.npy"
Y_PATH = "../data/1m_100/y_1m_100.npy"
CORE_COUNT = 8
TEST_RATIO = 0.2
# =====================================================================

print("Loading data...")
start_time = time.time()
X, y = load_data(X_PATH, Y_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=42)
load_time = time.time() - start_time
print(f"Data loaded in {load_time:.2f} s")

# =============================== CUSTOM MODEL ===============================
print("=" * 10 + " Custom Logistic Regression " + "=" * 10)
custom_model = LogisticRegressionParallel(CORE_COUNT)

start_time = time.time()
weights = custom_model.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Training time: {train_time:.2f} s")

start_time = time.time()
y_pred = custom_model.predict(X_test)
predict_time = time.time() - start_time

unique, counts = np.unique(y_pred, return_counts=True)
summary = {int(k): int(v) for k, v in zip(unique, counts)}
print(f"Predict summary: {summary}")
print(f"Prediction time: {predict_time:.2f} s")

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")
print(f"F1 score: {f1 * 100:.2f}%")

# =============================== SCIKIT-LEARN MODEL ===============================
print("=" * 10 + " Scikit-learn Logistic Regression " + "=" * 10)
sk_model = LogisticRegression(random_state=42, tol=1e-6, max_iter=30, n_jobs = max(1, CORE_COUNT))

start_time = time.time()
sk_model.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Training time: {train_time:.2f} s")

start_time = time.time()
y_pred = sk_model.predict(X_test)
predict_time = time.time() - start_time

unique, counts = np.unique(y_pred, return_counts=True)
summary = {int(k): int(v) for k, v in zip(unique, counts)}
print(f"Predict summary: {summary}")
print(f"Prediction time: {predict_time:.2f} s")

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")
print(f"F1 score: {f1 * 100:.2f}%")