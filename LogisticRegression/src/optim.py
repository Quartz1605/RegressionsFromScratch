import numpy as np

def gradients(X,w,b,y_true):

  X = np.asarray(X); y_true = np.asarray(y_true) 
  
  z = X@w + b
  
  y_pred = 1/(1 + np.exp(-z))
  y_pred = y_pred.reshape(-1,1)

  y_true = y_true.reshape(-1,1)
  
  n,d = X.shape

  grad_w = (1/n)*(X.T @ (y_pred - y_true))
  
  grad_b = (1/n)*np.sum(y_pred- y_true)

  return grad_w,grad_b



def gradients_multinomial(X,w,b,y_true):

  X = np.asarray(X) ; y_true = np.asarray(y_true)
  n,d = X.shape

  y_true = y_true.reshape(-1,1)

  unique_val = len(np.unique(y_true))

  y_loss = np.eye(unique_val)[y_true]

  

  grad_w = (1/n)*(X)

