import numpy as np

from .optim import gradients
from .metrics import mse


def predict_from_params(X,w,b):
  return X @ w + b


class LinearRegression:

  def __init__(self,lr=0.05,max_epochs=1000,tol=1e-8,verbose=False):
    self.lr = lr
    self.max_epochs = max_epochs
    self.tol = tol ## Tolerance bolte hai isko, what it does is basically tells you to stop GD when value reached till tol. 
    self.verbose = verbose ## Verbose here is for printing the loss val.
    self.w = None
    self.b = None
    self.history = []

  def fit(self, X, y):
    X = np.asarray(X); y = np.asarray(y)
    n, d = X.shape
    self.w = np.zeros(d)
    self.b = 0.0 
    last = np.inf
    
    for epoch in range(self.max_epochs):
      
      grad_w, grad_b, err = gradients(X, self.w, self.b,y)
      self.w -= self.lr * grad_w
      self.b -= self.lr * grad_b
      loss = mse(y, X @ self.w + self.b)
      self.history.append(loss)
      
      if self.verbose and (epoch % 100 == 0): 
        print(epoch, loss)
      if abs(last-loss) < self.tol: 
        break
      
      last = loss
    
    return self
  
  def predict(self,X):

    return predict_from_params(X,self.w,self.b)




  
    