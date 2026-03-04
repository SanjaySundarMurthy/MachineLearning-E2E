import numpy as np

class Linear_regression:
    def __init__(self,learning_rate,no_of_iterations):
        self.learning_rate=learning_rate
        self.no_of_iterations= no_of_iterations
    
    def fit(self,X,Y):
        self.m,self.n= X.shape
        
        ## y= mx+c
        ## initializing the weights and bias
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.y=Y
        
        for i in range(self.no_of_iterations):
            self.update_weights()
    
    def update_weights(self):
        y_prediction= self.predict(self.X)
        dw= -(2/self.m)*np.dot(self.X.T,(self.y-y_prediction))
        db= -(2/self.m)*np.sum(self.y-y_prediction)
        
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db
        
    def predict(self,X):
        return X.dot(self.w) + self.b