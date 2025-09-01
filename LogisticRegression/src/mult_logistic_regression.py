import numpy as np
from .optim import gradients


def calculate_softmax(z_matrix):

  z_matrix = z_matrix - np.max(z_matrix,axis=1,keepdims=True) #for numerical stability.Result doesn't change.

  exp_z = np.exp(z_matrix)

  return exp_z/np.sum(exp_z,keepdims=True,axis=1)


class MultinomialLogisticRegression:

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

    unique_val = len(np.unique(y_true))

    self.w = np.zeros(shape=(d,unique_val))
    
    self.b = np.zeros(shape=(unique_val))
    eps = 1e-9

    
    y_onehot = np.eye(unique_val)[y_true.reshape(-1)]

    for epoch in range(self.max_epochs):

      z_matrix = X @ self.w + self.b
      
      softmax_prob = calculate_softmax(z_matrix)

      y_val = np.sum(softmax_prob*y_onehot,axis=1)
      
      loss_val = np.mean(-np.log(y_val + eps))

      self.history.append(loss_val)

      dl_dz = softmax_prob - y_onehot

      dz_dw = (1/n)*(X.T @ dl_dz)
      dz_db = (1/n)*(np.sum(dl_dz,axis=0))

      self.w -= self.lr*dz_dw
      self.b -= self.lr*dz_db

      if abs(self.last-loss_val) < self.tol:
        break 
      
      if self.verbose and (epoch % 100 == 0):
        print(f"Epoch {epoch}: Loss = {loss_val:.4f}")


      self.last = loss_val

    return self
  
  def predict(self,X):

    z_matrix = X @ self.w + self.b

    y_softmax = calculate_softmax(z_matrix)

    y_pred = np.argmax(y_softmax,axis=1)

    return y_pred
  
  
  def predict_probab(self,X):

    z_matrix = X @ self.w + self.b

    y_softmax = calculate_softmax(z_matrix)

    return y_softmax





      



    







    

    