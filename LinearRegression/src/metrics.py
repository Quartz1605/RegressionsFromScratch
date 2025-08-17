import numpy as np

def mse(y_true,y_pred):

  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)

  return np.mean((y_true-y_pred)**2)