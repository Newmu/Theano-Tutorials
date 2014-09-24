import theano
"""theano.tensor contains functions for operations on theano variables 
similar interface to numpy operating on numpy arrays.
"""
from theano import tensor as T

import numpy as np

def floatX(X):
	"""
	Convert data to numpy arrays using the proper dtype specified by theano.
	"""
	return np.asarray(X,dtype=theano.config.floatX)

X = T.matrix()
Y = T.matrix()

data = np.linspace(-1,1,101)


