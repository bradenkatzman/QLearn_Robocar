# play.py

# play.py
import sys
sys.path.append('../')
from Game import car_game
import numpy as np
from NNs import Keras_NN

NUM_SENSORS = 3
REPORT_RATE = 10000

# main play method - looks similar to the neural net. We want to play indefinitely
def play(model):
	car_distance = 0
	game_state = car_game.GameState()

	# do nothing and get an initial state
	_, state = game_state.frame_step((2))

	# move indefinitely
	while True:
		car_distance += 1

		# choose an action by making a predicition and applying the greedy algorithm to the result
		action = (np.argmax(model.predict(state, batch_size=1)))

		# Take action
		_, state = game_state.frame_step(action)

		# report on the distance traveled at the specified frequency
		if car_distance % REPORT_RATE == 0:
			print("Current distance: %d frames." % car_distance)

# the first argument passed is the desired model to use for gameplay and the second and third define the kernel size to be used
def main(argv):
	saved_model = argv[0]
	model = Keras_NN(NUM_SENSORS, [int(sys.argv[1]), int(sys.argv[2])], saved_model)
	play(model)


if __name__ == "__main__":
	main(sys.argv[1:])