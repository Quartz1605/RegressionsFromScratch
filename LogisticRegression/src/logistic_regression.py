import numpy as np
from .optim import gradients

def predict_from_params(X,w,b):
  
  z =  X @ w + b

  return 1/((1 + np.exp(-z)))


class LogisticRegression:

  def __init__(self,lr=0.05,max_epochs=1000,tol=1e-8,verbose=False):
    self.lr = lr
    self.max_epochs = max_epochs
    self.tol = tol
    self.verbose = verbose
    self.w =  None
    self.b = None
    self.history = []
    self.last = np.inf

  
  def fit(self,X,y_true):

    X = np.asarray(X); y_true = np.asarray(y_true)

    n,d = X.shape

    self.w = np.zeros(shape=(d,1))
    
    self.b = 0.0
    eps = 1e-9

    y_true = y_true.reshape(-1,1)

    

    for epoch in range(self.max_epochs):
      

      grad_w,grad_b = gradients(X,self.w,self.b,y_true)

      self.w -= (self.lr)*(grad_w)
      self.b -= (self.lr)*(grad_b)

      y_pred = predict_from_params(X,self.w,self.b)
      y_pred = np.clip(y_pred,eps,1-eps)
      
      loss_val = -(1/n)*np.sum((y_true*np.log(y_pred))+((1-y_true)*np.log(1-y_pred)))

     

      self.history.append(loss_val)

      if self.verbose and (epoch % 100 == 0):
        print(f"Epoch {epoch}: Loss = {loss_val:.4f}")

      if abs(self.last-loss_val) < self.tol: 
        break

      self.last = loss_val

    return self


  def predict_probab(self,X):

    n = X.shape[0]

    z = X@self.w + self.b

    return 1/(1+np.exp(-z))
  
  def predict(self,X):

    n = X.shape[0]

    z = X@self.w + self.b

    y_pred = 1/(1+np.exp(-z))

    return (y_pred>=0.5).astype(int)
    


    

    







    