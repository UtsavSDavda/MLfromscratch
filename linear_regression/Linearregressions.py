import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = 0
    
    def predict(self, X):
        return X @ self.weights + self.bias

    def update_weights(self, X, Y):
        y_pred = X @ self.weights  
        dw = (-2 / self.m) * (X.T @ (Y - y_pred))  
        db = (-2 / self.m) * np.sum(Y - y_pred)    
        
        self.weights = self.weights - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db

    def fit(self, X, Y):
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)
        
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        
        for _ in range(self.epochs):
            self.update_weights(X, Y)