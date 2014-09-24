import theano
from theano import tensor as T
import numpy as np
from matplotlib import pyplot as plt

def floatX(X):
	return np.asarray(X,dtype=theano.config.floatX)

X = T.matrix() #symbolic input to the model
Y = T.matrix() #symbolic target to be predicted

W = theano.shared(floatX(np.random.randn(1,1)))
b = theano.shared(floatX(np.zeros((1))))

def model(X,W,b):
	return T.dot(X, W) + b

def cost_function(target, prediction):
	return T.mean(T.sqr(target-prediction))

def gradient_descent(parameters, gradients, step_size=0.01):
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

train = theano.function(inputs=[X, Y], outputs=error, updates=updates)

predict = theano.function(inputs=[X], outputs=prediction)

dataX = np.linspace(-1,1,101) # 101 data points spaced linearly between -1 and 1
dataY = 2*dataX + 1 + np.random.randn(*dataX.shape)*0.5 # Our slope is 2 and our y-intercept is 1

dataX = floatX(dataX).reshape(-1,1) # Convert to matrix format
dataY = floatX(dataY).reshape(-1,1) # Convert to matrix format

predicted = predict(dataX)

fig = plt.figure()
plt.scatter(dataX,dataY)
plt.plot(dataX,predicted,c='r')
plt.title('Random intial guess of model')
plt.show()

for i in range(1000): # 100 iterations through the data
	loss = train(dataX,dataY)
	if i % 100 == 0: print "iteration %.0f i, current error %.4f"%(i,loss)

predicted = predict(dataX)
plt.scatter(dataX,dataY)
plt.plot(dataX,predicted,c='r')
plt.title('Model predictions after trained')
plt.show()

print 'predicted slope',W.get_value()
print 'predicted y-intercept',b.get_value()

