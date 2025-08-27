import numpy as np

def binary_cross_entropy(y_pred,y_true):

  return -(y_true*np.log(y_pred))-((1-y_true)*np.log(1-y_pred))