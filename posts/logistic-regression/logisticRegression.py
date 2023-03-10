import numpy as np

class LogisticRegression: 
    
    def __init__(self):
        self.w = np.array([])
        self.loss_history = []
        self.score_history = []


    #Helper functions to perform gradient descent on the logistic loss
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def logistic_loss(self, y_hat, y):
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))

    def pad(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)
        
    def gradient(self, w, X, y):
        y_hat = X@w 
        return np.dot(self.sigmoid(y_hat) - y, X) / X.shape[0]

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def fit(self, X ,y, alpha=0.01, max_epochs=100):
        """
        Fits weights to data using standard gradient descent. There is not output value.
        This method updates the vector of weights, which is a an instance variable 
        of the class LogisticRegression().
        Input: 
            - X = Matrix composed of the data points 
            - y = Target vector 
            - alpha = learning rate 
            - max_epochs = maximum number of iterations 
        """
        X_ = self.pad(X)
        self.w = .5 - np.random.rand(X.shape[1] + 1)
        
        self.loss_history = [self.loss(X_, y)]
        self.score_history = [self.score(X_, y)]

        #Gradient descent 
        for _ in range(max_epochs):

            grad = self.gradient(self.w, X_, y)
            self.w -= alpha * grad

            self.loss_history.append(self.loss(X_, y))
            self.score_history.append(self.score(X_, y))

            #Convergence in gradient descent
            if np.isclose(self.loss_history[-1], self.loss_history[-2]):
                break 

    def predict(self, X):
        return 1 * (X @ self.w >= 0)

    def score(self, X, y):
        y_hat = self.predict(X)
        return (y_hat == y).mean()

    def loss(self, X, y):
        y_hat = X @ self.w
        return self.logistic_loss(y_hat, y).mean()


    def fit_stochastic(self, X, y, alpha=0.01, max_epochs=1000, batch_size=10, momentum = False):
        """
        Fits weights to data using stochastic gradient descent. There is not output value.
        This method updates the vector of weights, which is a an instance variable of
        the class LogisticRegression(). There is an optional parameter "momentum", a 
        method that improves the performance of stochastic gradient descent. 
        Input: 
            - X = Matrix composed of the data points 
            - y = Target vector 
            - alpha = learning rate 
            - max_epochs = maximum number of iterations 
            - batch_size = size of the batches used for stochastic gradient descent 
            - momentum = method to improve the performance of the descent. If it is True, then
            a variable beta is defined that affects the update of w
        """
        n = X.shape[0]

        self.w = .5 - np.random.rand(X.shape[1] + 1)
        beta = 0.9 if momentum else 0
        self.loss_history = [self.loss(self.pad(X), y)]  

        prev_w = 0 

        for j in np.arange(max_epochs):
            
            order = np.arange(n)
            np.random.shuffle(order)
            #Compute stochastic gradient
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X[batch,:]
                y_batch = y[batch]

                grad = self.gradient(self.w, self.pad(x_batch), y_batch) 

                temp = self.w
                self.w = self.w - (grad * alpha) + beta*(self.w - prev_w)
                prev_w = temp
            
            self.loss_history.append(self.loss(self.pad(X), y))
        
            if np.isclose(self.loss_history[-1], self.loss_history[-2]):
                break 
        