import numpy as np 

class Perceptron:
    
    def __init__(self):
        self.w = []
        self.history = []
    
    def fit(self, X, y, max_steps):
        "Fits wights to data until it has reached 'max_steps' iterations or the accuracy equals 1"
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        self.w = np.random.random(X.shape[1] + 1)
        y_ = 2*y - 1
        self.history = [self.score(X_, y)]

        for _ in range(max_steps):
            i = np.random.randint(X.shape[0])
            self.w += (y_[i] * (np.dot(X_[i], self.w)) < 0) * y_[i]*X_[i]
            self.history.append(self.score(X_, y))
            if self.history[-1] == 1:
                break 
 
    def predict(self, X):
        "Returns the model's prediction for the labels on the data"
        return 1*(np.matmul(X,self.w)>=0) 

    def score(self, X, y):
        "Returns the accuracy of the perceptron as a number from 0 to 1"
        y_ = self.predict(X)
        return (y_==y).mean()




    

        