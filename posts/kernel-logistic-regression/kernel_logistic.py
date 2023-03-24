import numpy as np 
from scipy.optimize import minimize


class KernelLogisticRegression:
    
    def __init__(self, kernel, **kernel_kwargs):
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs

    #Helper functions 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
    def pad(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def logistic_loss(self, y_hat, y):
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))

    def gradient(self, km, y):
        y_hat = self.v@km
        return (np.dot(self.sigmoid(y_hat) - y, km)) / km.shape[0]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   

    def fit(self, X, y):
        self.X_train = X
        X_ = self.pad(X)

        v0 = np.random.rand(X.shape[0])
        km = self.kernel(X_, X_, **self.kernel_kwargs)
        
        def empirical_risk(X, y, v, loss):
            y_hat = X@v
            return loss(y_hat, y).mean()
        
        result = minimize(lambda v: empirical_risk(km, y, v, self.logistic_loss), x0 = v0) 
        self.v = result.x
        
    def predict(self, X):
        km = self.kernel(X, self.X_train, **self.kernel_kwargs)
        return (np.matmul(km, self.v) > 0).astype(int)

    def score(self, X, y):
        y_ = self.predict(X)
        return (y_ == y).mean() 


    # def minimize(self, km, y, alpha = 0.1, max_epochs = 1000):
    #     for _ in range(max_epochs):

    #         grad = self.gradient(km, y)
    #         self.v -= alpha * grad

    #         #Convergence
    #         if np.allclose(grad, np.zeros(len(grad))):
    #             break
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # def fit(self, X, y):
    #     self.X_train = X
    #     X_ = self.pad(X)

    #     km = self.kernel(X_, X_, **self.kernel_kwargs)
    #     self.v = np.random.rand(X.shape[0])
    #     #Empirical Risk Minimization 
    #     self.minimize(km, y)


    










    
        


