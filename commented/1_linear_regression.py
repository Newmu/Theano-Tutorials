import theano
from theano import tensor as T
"""theano.tensor contains functions for operations on theano variables 
similar interface to numpy operating on numpy arrays.
"""

import numpy as np
from matplotlib import pyplot as plt

def floatX(X):
	"""
	Theano works on specific data types such as float32 or float64 and will complain 
	if you pass it data types that it does not expect. floatX converts data to numpy 
	arrays with proper dtype specified by theano configuration.
	"""
	X = np.asarray(X,dtype=theano.config.floatX)
	return X

"""
Theano functions are definied in terms of operations on symbolic variables.
There are variety of symbolic variable types depending on the dimensionality of the data.
T.scalar is for variables that are 0-dimensional ndarrays - single numbers such as 0.5
T.vector is for variables that are 1-dimensional ndarrays - lists such as [0.1,-0.1]
T.matrix is for variables that are 2-dimensional ndarrays - matrices (list of list) such as [[1.,0],[0.,1.]]
T.tensor3 is for variables that are 3-dimensional ndarrays - list of 2d images grayscale images
T.tensor4 is for variables that are 4-dimensional ndarrays - list of color images
For generalization purposes we will define linear regression in terms of matrix operations.
"""

X = T.matrix() #symbolic input to the model
Y = T.matrix() #symbolic target to be predicted

"""
Theano.shared is allows us to define hybrid symbolic/real variables that can be used
in symbolic functions and also have actual parameterizations at run time that can be learned.
We instantiate our weight matrix (W) to a randomly sampled number and our bias (b) to zero.
"""
W = theano.shared(floatX(np.random.randn(1,1)))
b = theano.shared(floatX(np.zeros((1))))

def model(X,W,b):
	"""
	Linear regression in terms of matrix multiplication.
	W is effectively our "slope" while b is our "y-intercept" we have just written it in a 
	general way so we can have multiple slopes for multiple inputs and multiple offsets 
	for multiple outputs.

	X is our symbolic theano variable representing input.
	W is a weight matrix we will learn.
	b is our bias allowing us to learn offsets.
	"""
	prediction = T.dot(X, W) + b
	return prediction

def cost_function(target, prediction):
	"""
	In a gradient based learning framework, we need an error to calculate the gradients
	of parameters with respect to. For a regression problem - predicting a real value
	a standard error function is the squared error. Theano's gradient function works 
	with respect to scalars, so we need to take the mean of the squared error to take 
	a error vector/matrix and convert it to a scalar.

	target is the desired output you are trying to predict with your model.
	prediction is the prediction the model will make.
	"""
	squared_error = T.sqr(target-prediction)
	mean_squared_error = T.mean(squared_error)
	return mean_squared_error

def gradient_descent(parameters, gradients, step_size=0.01):
	"""
	In order to "learn" our model, we define a function which takes parameters and gradients
	of those parameters and provides an update rule to modify the parameters based on the
	gradients. Since our gradients our with respect to an error which we are trying to minimize
	descending the gradient corresponds to minimizing that error.

	parameters is a list of the parameters we are learning
	gradients is a list of gradients of those parameters
	step_size controls how "quickly" to move in the desired direction. Large values may allow
	the model to learn quicker but may become unstable due to overshooting.
	"""
	updates = []
	for p,g in zip(parameters, gradients):
		updated_p = p - step_size * g
		updates.append((p,updated_p))
	return updates

prediction = model(X,W,b)
error = cost_function(Y, prediction)

parameters = [W, b] # A list of parameters to be learned by our model
gradients = T.grad(cost=error, wrt=parameters) #Gradients of our parameters with respect to our error.
updates = gradient_descent(parameters,gradients) #Function to be called to update our model

"""
theano.function takes our so far symbolic functions and compiles them to
callable python functions that can work on real data.
In this case we are creating a function which will take in two numpy data arrays
and substitute them for the theano variables X, the input to the model and Y,
the desired output.
outputs is what we want to compute with this function, in this case the error.
updates is an expression that allows us to update the models parameters on a given call to train.
"""
train = theano.function(inputs=[X, Y], outputs=error, updates=updates)

"""
We want to be able to have our model predict on data. This is the theano function to do that.
"""
predict = theano.function(inputs=[X], outputs=prediction)

"""
Now that we have a training function we should train our model on some data.
"""

dataX = np.linspace(-1,1,101) # 101 data points spaced linearly between -1 and 1
dataY = 2*dataX + 1 # Our slope is 2 and our y-intercept is 1

dataY += np.random.randn(*dataY.shape)*0.5 # Adding some noise

dataX = floatX(dataX).reshape(-1,1) # Convert to matrix format
dataY = floatX(dataY).reshape(-1,1) # Convert to matrix format
 
"""
Let's see what our data looks like and what our model predicts after random initialization.
"""
predicted = predict(dataX)

fig = plt.figure()
plt.scatter(dataX,dataY)
plt.plot(dataX,predicted,c='r')
plt.title('Random initial guess of model')
plt.show()

for i in range(1000): # 100 iterations through the data
	loss = train(dataX,dataY)
	if i % 100 == 0: print "iteration %.0f i, current error %.4f"%(i,loss)

"""
Now let's see what the model predicts after being trained
"""
predicted = predict(dataX)
plt.scatter(dataX,dataY)
plt.plot(dataX,predicted,c='r')
plt.title('Model predictions after trained')
plt.show()

print 'predicted slope',W.get_value()
print 'predicted y-intercept',b.get_value()
