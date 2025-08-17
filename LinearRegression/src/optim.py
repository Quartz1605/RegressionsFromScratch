import numpy as np

def gradients(X,w,b,y):
  
  n = X.shape[0]
  y_pred = X @ w + b

  err = y_pred - y

  grad_w = (2/n)*(X.T @ err)
  grad_b = (2/n)*np.sum(err)

  return grad_w,grad_b,err


  
  