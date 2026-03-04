import numpy as np

class Logistic_Regression:
    def __init__(self, learning_rate, no_of_iteration):
        self.learning_rate = learning_rate
        self.no_of_iteration = no_of_iteration
    
    def fit(self,X,Y):
        ## no of samples , No of features
        self.m, self.n= X.shape
        self.weights= np.zeros(self.n)
        self.bias=0
        self.X= X
        self.Y=Y
        for _ in range(self.no_of_iteration):
            self.update_weights()
    
    def update_weights(self):
        linear_model= np.dot(self.weights,self.X) + self.bias
        y_hat= 1.0 / (1 + np.exp(-linear_model))
        ## Find the derivatives
        dw = (1/self.m)*np.dot((y_hat-self.Y),self.X.T)
        db= (1/self.m)*np.sum(y_hat-self.Y)
        
        self.weights= self.weights - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db
    
    def predict(self,X):
        ## sigmoid activation function 
        linear_model= np.dot(self.weights,X) + self.bias
        y_pred= 1.0 / (1 + np.exp(-linear_model))
        y_pred= np.where(y_pred > 0.5 ,1 ,0)
        return y_pred