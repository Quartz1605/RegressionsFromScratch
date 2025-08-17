import numpy as np

def generate_X(n,d):

  rng = np.random.default_rng(42)

  X = rng.normal(size=(n,d))

  return X

