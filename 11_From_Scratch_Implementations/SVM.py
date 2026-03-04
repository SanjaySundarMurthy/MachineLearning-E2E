import numpy as np

class svm_classifier:
    def __init__(self,learning_rate,no_of_iterations,lambda_params):
        self.learning_rate= learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_params = lambda_params
    
    def fit(self,X,Y):
        # no of datapoints and no of features input
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        self.X=X
        self.Y=Y
        
        for _ in range(self.no_of_iterations):
            self.update_weights()
    
    def update_weights(self):
        # label encoding
        y_label = np.where(self.Y <= 0,-1,1)
        for index , X_i in enumerate(self.X):
            condition = y_label[index]* (np.dot(self.weights,X_i) - self.bias) >= 1
            if condition == True:
                dw =  2* self.lambda_params * self.weights
                db =  0
            else:
                dw = 2 *  self.lambda_params* self.weights - np.dot(X_i, y_label[index])
                db = y_label[index]
            
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.weights -  self.learning_rate * db
      
    def predict(self,X):
        output = np.dot(X,self.weights) - self.bias
        predicted_labels= np.sign(output)
        y_hat= np.where(predicted_labels <= -1,0,1)
        return y_hat