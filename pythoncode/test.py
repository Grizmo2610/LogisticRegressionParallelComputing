from model import LogisticRegressionParallel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
import json

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

def bechmark_hepler(X_train, X_test, y_train, y_test,
                    core: int, mt: int = 0, 
                    display: bool = True, override: bool = False,
                    path="runs/data.json") -> dict:
    """
    Run and record benchmark results for a logistic regression model.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Training and testing feature matrices.
    y_train, y_test : np.ndarray
        Training and testing labels.
    core : int
        Number of CPU cores used for training.
    mt : int, default=0
        Model type. 
        0 = LogisticRegressionParallel (custom implementation).
        1 = Standard LogisticRegression from sklearn.
    display : bool, default=True
        Whether to print intermediate benchmark information.
    override : bool, default=False
        If True, overwrite the existing JSON file instead of appending.
    path : str, default="runs/data.json"
        File path to store benchmark results.

    Returns
    -------
    dict
        Dictionary containing benchmark results:
        {id, core, features, samples, accuracy, f1, train_time, predict_time, summary}
    """

    # Get folder name and create directory if needed
    folder_name = os.path.basename(os.path.dirname(path))
    if folder_name and not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    # Select model type
    if mt == 0:
        model = LogisticRegressionParallel(core)
    elif mt == 1:
        model = LogisticRegression(random_state=42, tol=1e-6, max_iter=30, n_jobs=max(1, core))
    else:
        raise ValueError("Unidentified model type")

    # Train model and record time
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    if display:
        print(f"Training time: {train_time:.2f} s")

    # Prediction phase
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start

    # Summary of predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    summary = {int(k): int(v) for k, v in zip(unique, counts)}
    if display:
        print(f"Predict summary: {summary}")
        print(f"Prediction time: {predict_time:.2f} s")

    # Accuracy and F1 score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    if display:
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"F1 score: {f1 * 100:.2f}%")

    # Prepare benchmark record
    id = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "id": id,
        "core": core,
        "features": len(X_train[0]),
        "samples": len(X_train),
        "accuracy": acc,
        "f1": f1,
        "train_time": train_time,
        "predict_time": predict_time,
        "summary": summary
    }

    # Read existing file and append or overwrite
    if os.path.exists(path) and not override:
        with open(path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        if isinstance(old_data, list):
            old_data.append(data)
        else:
            old_data = [old_data, data]
    else:
        old_data = [data]

    # Write updated data to JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old_data, f, ensure_ascii=False, indent=2)

    return data

    
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

result = bechmark_hepler(X_train, X_test, y_train, y_test, core=CORE_COUNT, mt=0, display=True, override=True)
print(result)