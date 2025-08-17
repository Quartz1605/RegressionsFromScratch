import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import numpy as np
from tests.test_sanity import generate_X


X = generate_X(10,3)
#print(X)


def setup_linear_regression(b_true,X,seed,w_true=None):

  rng = np.random.default_rng(seed) #Isme "rng" stands for random number generator,yahape u are making a object of it.

  data_shape = X.shape

  if w_true is None:
    w_true = rng.normal(size=data_shape[1])

  y =  X @ w_true + b_true + rng.normal(0,1.0,size=data_shape[0])

  return y,w_true,b_true

y,w,b = setup_linear_regression(b_true=1.0,X=X,seed=42)
print(y)
print(w)
print(b)

