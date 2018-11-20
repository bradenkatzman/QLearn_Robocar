# QLearn.py
import sys
sys.path.append('../')
from Game import car_game
import numpy as np
import random
import csv
from NNs import Keras_NN
import os.path
import timeit

NUM_INPUT = 3
GAMMA = 0.9
TUNING = False
NUM_OBSERVATION_FRAMES_BEFORE_TRAINING = 1000
TRAIN_FRAMES = 100000
EPS = 1 # start by strictly exploring
SAVE_RATE = 25000 # the frequency with which we want to save the state of the model

def train_net(model, params):

	filename = params_to_filename(params)

	EPSILON = EPS

	batchSize = params['batchSize']
	buffer = params['buffer']

	max_car_distance = 0
	car_distance = 0
	t = 0
	data_collect = []
	replay = [] # this will store tuples (STATE, ACTION, REWARD, STATE^)

	loss_log = []

 	# Create a new game instance
	game_state = car_game.GameState()

 	# get initial state by performing no action and getting the state
	_, state = game_state.frame_step((2))

 	# time it for efficiency purposes
	start_time = timeit.default_timer()

	# do a training run
	while t < TRAIN_FRAMES:
		t += 1 # keep track of the time
		car_distance += 1 # keep track of the total distance traveled

 		# make an action choice, either EXPLORE or EXPLOIT
		if random.random() < EPSILON or t < NUM_OBSERVATION_FRAMES_BEFORE_TRAINING:
			action = np.random.randint(0, 3) # random
		else:
 			# get Q values for each action --> a probability distrubition across the action choices
			qval = model.predict(state, batch_size=1)

 			# use the greedy algorithm to pick the best action
			action = (np.argmax(qval))

 		# take the action and figure out the reward it confers and the change it state it gives
		reward, new_state = game_state.frame_step(action)

 		# experience replay storage --> add the tuple of four 4 to the list
		replay.append((state, action, reward, new_state))

 		# if we've finished the observation stage (where we don't EXPLOIT but only EXPLORE), let's start training
		if t > NUM_OBSERVATION_FRAMES_BEFORE_TRAINING:

 			# if the size of the buffer is filling up, we'll pop off the oldest tuple
			if len(replay) > buffer:
				replay.pop(0)

 			# take a random sample (of size batchSize) from the experience replay to create a minibatch
			minibatch = random.sample(replay, batchSize)

 			# get the training values from this minibatch
			X_train, y_train = process_minibatch(minibatch, model)

 			# train the model on this batch, and record the LossHistory
			history = Keras_NN.LossHistory()
			model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0, callbacks=[history])
			loss_log.append(history.losses)

 		# update the state to the new state (as a result of the action taken)
		state = new_state

 		# decay epsilon over time
		if EPSILON > 0.1 and t > NUM_OBSERVATION_FRAMES_BEFORE_TRAINING:
			EPSILON -= (1.0/TRAIN_FRAMES)

 		# now we need to check if we died as a result of the action taken
		if reward == -500:
 			# log the car's distance at this T
			data_collect.append([t, car_distance])

 			# update the maximum distance travelled in a game if this distance is greater
			if car_distance > max_car_distance:
				max_car_distance = car_distance


 			# see how long the game lasted, and the number of frams processed per second
			total_time = timeit.default_timer() - start_time
			fps = car_distance/total_time

 			# output to keep track of some values
			print("Max: %d at %d\tEPSILON %f\t(%d)\t%f fps" %
				(max_car_distance, t, EPSILON, car_distance, fps))

 			# reset counter values
			car_distance = 0
			start_time = timeit.default_timer()


 		# save the model if it is time
		if t % SAVE_RATE == 0:
			model.save_weights('../saved-models/' + filename + '-' + str(t) + '.h5', overwrite=True)
			print("Saving model %s - %d" % (filename, t))


 	# all of the frames have been processed so let's log the results
	log_results(filename, data_collect, loss_log)



def log_results(filename, data_collect, loss_log):
	with open('../results/sonar-frames/learn_data-' + filename + '.csv', 'w') as data_dump:
		wr = csv.writer(data_dump)
		wr.writerows(data_collect)

	with open('../results/sonar-frames/loss_data-' + filename + '.csv', 'w') as lf:
		wr = csv.writer(lf)
		for loss_item in loss_log:
			wr.writerow(loss_item)


def process_minibatch(minibatch, model):
	mb_len = len(minibatch)

	old_states = np.zeros(shape=(mb_len, 3))
	actions = np.zeros(shape=(mb_len,))
	rewards = np.zeros(shape=(mb_len,))
	new_states = np.zeros(shape=(mb_len, 3))

	for i, m in enumerate(minibatch):
		# extract the tuple
		old_state_m, action_m, reward_m, new_state_m = m

		old_states[i, :] = old_state_m[...]
		actions[i] = action_m
		rewards[i] = reward_m
		new_states[i, :] = new_state_m[...]


	# calculate the q values over the state sets from before and after actions
	old_qvals = model.predict(old_states, batch_size=mb_len)
	new_qvals = model.predict(new_states, batch_size=mb_len)

	maxQs = np.max(new_qvals, axis=1)
	y = old_qvals

	# the indices of the actions where the car did not crash
	non_term_inds = np.where(rewards != -500)[0]

	# the indices of the actions where the car did crash
	term_inds = np.where(rewards == -500)[0]

	# set the rewards to the training values, magnifying the non-crash rewards by Q
	y[non_term_inds, actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (GAMMA * maxQs[non_term_inds])
	y[term_inds, actions[term_inds].astype(int)] = rewards[term_inds]

	X_train = old_states
	y_train = y
	return X_train, y_train

def params_to_filename(params):
	return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + str(params['batchSize']) + '-' + str(params['buffer'])

def launch_learn(params):
	filename = params_to_filename(params)
	print("Trying %s" % filename)
	if not os.path.isfile('../results/sonar-frames/loss_data-' + filename + '.csv'):
		open('../results/sonar-frames/loss_data-' + filename + '.csv', 'a').close()
		print("Starting test.")

		# train the model
		model = Keras_NN.simple_neural_net(NUM_INPUT, params['nn'])
		train_net(model, params)
	else:
		print("Already tested")


# RUNNING CODE
if __name__ == "__main__":
	if TUNING:
		param_list = []
		nn_params = [[164, 150], [256, 256],
					[512, 512], [1000, 1000]]
		batchSizes = [40, 100, 400]
		buffers = [10000, 50000]

		for nn_param in nn_params:
			for batchSize in batchSizes:
				for buffer in buffers:
					params = {"batchSize": batchSize, "buffer": buffer, "nn": nn_param}
					param_list.append(params)

		for param_set in param_list:
			launch_learn(param_set)
	else:
		nn_param = [128, 128]
		params = {"batchSize": 64, "buffer": 50000, "nn": nn_param}
		model = Keras_NN.simple_neural_net(NUM_INPUT, nn_param)
		train_net(model, params)