import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.01,epochs=10,threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
    
    def fit(self, X, Y):
        X  = np.array(X)
        Y = np.array(Y)
        self.row, self.col = X.shape
        self.weights = np.zeros(self.col)
        self.bias = 0
        for _ in range(self.epochs):
            self.update_weights(X,Y)

    def update_weights(self,X,Y):
        Y_prob =  1/(1 + np.exp(-(X@self.weights + self.bias)))
        dw = (1/self.row)*(X.T@(Y_prob - Y))
        db = (1/self.row)*np.sum(Y_prob - Y)
        self.weights = self.weights - (self.learning_rate)*(dw)
        self.bias = self.bias - (self.learning_rate)*(db)
    
    def predict(self, Z):
        Y_prob =  1/(1 + np.exp(-(Z@self.weights + self.bias)))
        return (Y_prob >= self.threshold).astype(int)