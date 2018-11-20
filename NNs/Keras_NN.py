# Keras_NN.py

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM # probably won't use this
from keras.callbacks import Callback 

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = [] # set up the array to record the losses after each batch of training data runs through the net

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

# Fully connected 3 layer neural net that uses rectified linear activation units after input and hidden layer
def simple_neural_net(num_sensors, parameters, load=''):
	model = Sequential() # the sequential model is a linear stack of layers (i.e. the structure of the neural network that is feedforward). Layers are added with model.add()

	# add the first layer i.e. the input layer. It is a dense layer that uses the LeCun Uniform initializer, which draws samples from a uniform distribution within [-limit, limit] where limit
	# is sqrt(3 / fan_in) where fan_in is the number of input units in the weight tensor i.e. num_sensors
	model.add(Dense(parameters[0], init='lecun_uniform', input_shape=(num_sensors,)))

	# add a rectified linear activition unit to the end of the first layer. This unit has output 0 if the input is less than 0, and raw output otherwise i.e. it is a positive activation function
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	# add the second layer i.e. the hidden layer. It is another dense layer with rectified linear activitation unit after
	model.add(Dense(parameters[1], init='lecun_uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	# add the third layer i.e. the output layer. This time just use a linear activation
	model.add(Dense(3, init='lecun_uniform'))
	model.add(Activation('linear'))

	# make an RMSProp optimizer for the model. Default values will be used for the learning rate, rho, epsilon, and the decay
	rms = RMSprop()
	model.compile(loss='mse', optimizer=rms)

	if load:
		model.load_weights(load)

	return model