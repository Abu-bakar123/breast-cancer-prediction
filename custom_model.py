import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []

    # ------------------------------------------------------------
    def _sigmoid(self, z):
        """Convert raw scores into probabilities."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # ------------------------------------------------------------
    def _cost(self, y_true, y_pred):
        """Binary cross-entropy loss."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        return -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

    # ------------------------------------------------------------
    def _gradient(self, X, y_true, y_pred):
        """Calculate gradients for weights and bias."""
        m = X.shape[0]
        error = y_pred - y_true

        dW = (X.T @ error) / m
        db = np.sum(error) / m

        return dW, db

    # ------------------------------------------------------------
    def fit(self, X, y):
        """Train the model using batch gradient descent."""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):
            # 1. Forward pass
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            # 2. Calculate and store loss
            loss = self._cost(y, y_pred)
            self.loss_history.append(loss)

            # 3. Calculate gradients
            dW, db = self._gradient(X, y, y_pred)

            # 4. Update weights and bias
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

            if self.verbose and i % 100 == 0:
                print(f"Iter {i:4d} | loss = {self.loss_history[-1]:.4f}")

        return self

    # ------------------------------------------------------------
    def predict_proba(self, X):
        """Return probability of class 1 for each row."""
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    # ------------------------------------------------------------
    def predict(self, X, threshold=0.5):
        """Return predicted class labels: 0 or 1."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int) 