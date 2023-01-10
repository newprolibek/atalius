import numpy as np
from preproccessing import batches
import functions
'''
TODO list:

- Regularization (l1, l2, elastic)
- Normalization (minmax and standardscaler)
- Earlystopping
- Evaluation methods
- Stochastic GD

- Parent Class for linear models

- Logistic Regression

'''

class LinearRegression:
    
    def __init__(self):
        self.w = None

    def fit(self, X_, y_, pseudo_inverse = False, lr=0.0001, epochs=100, batch_size=None):

        X = np.array(X_)

        if(len(X.shape) == 1):
            X = X.reshape(X.shape[0], 1) # if X is an array, we convert it to column-vector
        
        y = np.array(y_)
        y = y.reshape(y.shape[0], 1) # making a column-vector
        
        n, k = X.shape

        if self.w is None:
            self.w = np.zeros((k+1, 1))
        X = np.hstack((np.ones((n, 1)), X))

        if not pseudo_inverse:

            l_hist = []
            w_hist = []

            for i in range(epochs):
                for X_batch, y_batch in batches(X, y, batch_size): # stochastic gradient method

                    y_pred = X_batch @ self.w
                    gradient = (1 / n) * X_batch.T @ (y_pred - y_batch) # compute gradient of MSE which equals (w*x+b-y)*x
                    # in matrix form it*s X^T*(y_hat - y)
                    self.w -= lr * gradient # gradient descent method
                    w_hist.append(self.w)

            return l_hist, w_hist
        
        else:
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
            return self.w

    def predict(self, X):
        X = np.array(X)
        
        if(len(X.shape) == 1):
            X = np.reshape(X.shape[0], 1)
        
        X_tmp = np.hstack((np.ones((X.shape[0], 1)), X))

        return X_tmp @ self.w


class LogisticRegression:
    def __init__(self):
        self.w = None

    def fit(self, X_, y_, epochs=100, lr=0.0001, batch_size=None):
        
        X = np.array(X_)
        if(len(X) == 1):
            X = X.reshape()
        n, k = X.shape[0]
        X = np.hstack((np.ones(n, 1), X))

        y = np.array(y_)
        y = y.reshape()

        if self.w is None:
            self.w = np.random.randn((k+1, 1))

        w_hist = []
        l_hist = []

        for i in range(epochs):
            for X_batch, y_batch in batches(X, y, batch_size):

                y_pred = functions.sigmoid(X @ self.w)
                gradient = X_batch.T @ (y_pred - y)
                self.w -= lr * gradient

        return l_hist, w_hist

    def predict(self, X_):
        X = np.array(X_)
        
        if(len(X.shape) == 1):
            X = np.reshape(X.shape[0], 1)
        
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        return X @ self.w