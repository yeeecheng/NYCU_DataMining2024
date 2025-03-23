import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, regularization="l2"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization  # None, 'l1', 'l2', or 'elastic_net'
        self.weights = None
        self.bias = None
        self.best_weights = None
        self.best_bias = None
        self.best_val_loss = float('inf')
    
    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.best_weights = self.weights.copy()
        self.best_bias = self.bias

        patience = 20000
        no_improve_count = 0
        lambda_l1 = 0.01
        lambda_l2 = 0.01

        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            train_loss = self.rmse(y, y_predicted)
            
            
            # Apply Regularization
            if self.regularization == "l1":
                # L1 Regularization gradient
                dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y)) + lambda_l1 * np.sign(self.weights)
                db = (2 / n_samples) * np.sum(y_predicted - y)
            elif self.regularization == "l2":
                # L2 Regularization gradient
                dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y)) + 2 * lambda_l2 * self.weights
                db = (2 / n_samples) * np.sum(y_predicted - y)
            elif self.regularization == "elastic_net":
                # Elastic Net gradient (L1 + L2)
                dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y)) + lambda_l1 * np.sign(self.weights) + 2 * lambda_l2 * self.weights
                db = (2 / n_samples) * np.sum(y_predicted - y)
            elif self.regularization == "normal":
                # normal
                dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
                db = (2 / n_samples) * np.sum(y_predicted - y)


            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            self.train_losses.append(train_loss)

            if X_val is not None and y_val is not None:
                val_loss = self.score(X_val, y_val)
                self.val_losses.append(val_loss)

                if (epoch + 1) % 1000 == 0:
                    print(f"epoch {epoch + 1}, train loss: {train_loss}, val loss: {val_loss}, best val loss: {self.best_val_loss}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_weights = self.weights.copy()
                    self.best_bias = self.bias
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return self.rmse(y, y_pred)
    
    def predict_validation(self, X):
        if self.best_weights is not None and self.best_bias is not None:
            return np.dot(X, self.best_weights) + self.best_bias
        else:
            return self.predict(X)
        
    def save_weights(self, filepath="weights.npz"):
        """Save weights and bias to a file"""
        np.savez(filepath, weights=self.best_weights, bias=self.best_bias)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath="weights.npz"):
        """Load weights and bias from a file"""
        try:
            data = np.load(filepath)
            self.best_weights = data['weights']
            self.best_bias = data['bias']
            self.weights = self.best_weights.copy()
            self.bias = self.best_bias
            print(f"Weights loaded from {filepath}")
        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
        except KeyError:
            print("Error: Invalid file format. Ensure it's a correct weights file.")



