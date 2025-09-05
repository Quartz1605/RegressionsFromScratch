import numpy as np
import pandas as pd

def calculate_prior_probabilities(data,zero_X,one_X):

  total_obs = data.shape[0]

  one_probab = one_X.shape[0]/total_obs
  zero_probab = zero_X.shape[0]/total_obs

  return one_probab,zero_probab


def data_preprocessing(data):

  zero_data = data[data[:,-1] == 0]
  one_data = data[data[:,-1] == 1]

  zero_X = zero_data[:,:-1]
  one_X = one_data[:,:-1]

  return zero_X,one_X

def log_gaussian_feature_density(mean_data,variance_data,X_data):

  log_gaussian_val = np.sum(((-1/2)*np.log(2*np.pi*variance_data)) - (((X_data-mean_data)**2)/(2*variance_data)),axis=1)

  return log_gaussian_val


class GaussianNaiveBayes:

  def __init__(self,verbose=False):
    self.epsilon = 0.01
    self.prior_one = None
    self.prior_zero = None

    self.mean_one_X = None
    self.mean_zero_X = None

    self.var_one_X = None
    self.var_zero_X = None

    self.log_gauss_one_val = None

  
  def fit(self,X_true,y_true):

    data = np.concatenate((X_true,y_true.reshape(-1,1)),axis=1)

    zero_X,one_X = data_preprocessing(data)
    
    one_probab,zero_probab = calculate_prior_probabilities(data,zero_X,one_X)
    self.prior_one = np.log(one_probab)
    self.prior_zero = np.log(zero_probab) 

    self.mean_one_X = one_X.mean(axis=0)
    self.mean_zero_X = zero_X.mean(axis=0)

    self.var_one_X = one_X.var(axis=0) + self.epsilon
    self.var_zero_X = zero_X.var(axis=0) + self.epsilon

    print("Model fitted")

    return self

  def predict(self,X):

    X = np.asarray(X)

    log_density_one = log_gaussian_feature_density(self.mean_one_X,self.var_one_X,X)
    log_density_zero = log_gaussian_feature_density(self.mean_zero_X,self.var_zero_X,X)

    one_probab = self.prior_one + log_density_one
    zero_probab = self.prior_zero + log_density_zero

    return (one_probab > zero_probab).astype(int)
  
  




    






    

    
