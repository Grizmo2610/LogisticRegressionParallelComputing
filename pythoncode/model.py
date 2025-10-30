import os
import platform

if platform.system().lower() == "windows":
    # Add MinGW bin directory to DLL search path on Windows
    # Change the path below if MinGW is installed in a different location
    os.add_dll_directory(r"C:\mingw64\bin")

import Logistic

class LogisticRegressionParallel:
    """
    Python wrapper for a C++ Logistic Regression model
    with optional parallel execution across CPU cores.
    """

    def __init__(self, core: int):
        """
        Initialize the Logistic Regression model from the C++ module.

        Args:
            core (int):
                - Number of CPU cores to use for parallel training.
                - If core < 0, training runs sequentially (no parallelization).

        Attributes:
            model: Instance of Logistic.LogisticRegression from the C++ backend.
            weights (list[float]): Model weights after training.
        """
        self.model = Logistic.LogisticRegression(core)
        self.weights = []

    def fit(self, X, y, lr: float = 1e-2, epsilon: float = 1e-6, maxtier: int = 30) -> list[float]:
        """
        Train the logistic regression model.

        Args:
            X (list[list[float]]): Feature matrix (samples × features).
            y (list[int]): Target labels (0 or 1).
            lr (float): Learning rate.
            epsilon (float): Convergence threshold.
            maxtier (int): Maximum number of training iterations.

        Returns:
            list[float]: Trained model weights.
        """
        self.weights = self.model.fit(X, y, lr, epsilon, maxtier)
        return self.weights

    def predict(self, X, thresh: float = 0.5) -> list[int]:
        """
        Predict class labels for input data.

        Args:
            X (list[list[float]]): Feature matrix for prediction.
            thresh (float): Decision threshold for classification (default 0.5).

        Returns:
            list[int]: Predicted class labels (0 or 1).
        """
        return self.model.predict(X, thresh)