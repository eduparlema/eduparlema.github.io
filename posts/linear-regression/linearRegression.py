import numpy as np 

class LinearRegression:

    def __init__(self):
        self.w = np.array([])
        self.score_history = []


    def pad(self,X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    
    def fit_analytical(self, X, y):
        X_ = self.pad(X)
        self.w = np.linalg.inv(X_.T@X_)@X_.T@y

    def fit_gradient(self, X, y, alpha = 0.0001, max_epochs = 100):
        X_ = self.pad(X)
        self.w = np.random.rand(X_.shape[1])
        self.score_history = [ self.score(X, y) ]

        P = X_.T @ X_
        q = X_.T @ y

        for _ in range(max_epochs):

            grad = P @ self.w  -  q 
            self.w -= alpha * grad 

            self.score_history.append(self.score(X, y))

            #Convergence
            if np.allclose(grad, np.zeros(len(grad))):
                break


    def predict(self, X):
        X = self.pad(X)
        return X @ self.w 

    def score(self, X, y):
        y_ = self.predict(X)
        y_bar = y.mean()
        return 1 - (((y_ - y)**2).sum() / ((y_bar - y)**2).sum())



    


    


