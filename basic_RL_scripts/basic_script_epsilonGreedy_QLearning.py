import sys
import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

# this is a small simulation of a 10 slot machine problem where the goal is to get the highest reward
# scripts adapted from: http://outlace.com/rlpart1.html

# we'll use the epsilon-greedy algorithm to choose actions (static epsilon value)

n = 10 # 10 slot machines
arms = np.random.rand(n) # makes a numpy array with 10 (n) floating-point values that correspond to probabilities of each slot-machine
epsilon = 1 # the starting value
action_value_memory = np.array([np.random.randint(0, (n+1)), 0]).reshape(1, 2) # initialize a memory array that will store key-value pairs action and their corresponding values (reward)
				# makes an array that has 1 row that is the indices of the actions [0, n+1) in random order and zeros corresponding to their values (actions start with 0 reward because it is computed at runtime i.e. EXPLORATION)

# each arm has a probability. we'll set the reward for each arm by looping to 10, and at each step 
# doing the following algorithm
def reward(prob):
	reward = 0
	for i in range(10):
		if random.random() < prob: # if a random number [0,1] is less than the random probability assigned to this arm, incrememnt the reward value
			reward += 1
	return reward # note that the maximum reward is 10

# greedy algorithm to select the best action based on the memory array
def bestAction(actions):
	bestAction = 0
	bestMean = 0
	for action in actions:
		avg = np.mean(actions[np.where(actions[:,0] == action[0])][:, 1]) # calculate the mean reward for each action by finding the matching action index and gathering all previous recorded rewards from that action
		if bestMean < avg: # if the average reward for this action is better than any other average rewards so far, set this action and its reward value as the flagged values
			bestMean = avg
			bestAction = action[0]
	return bestAction



###############################################################################################
# main play loop
fig = plt.figure()
plt.xlabel("Plays")
plt.ylabel("Avg Reward")
for iteration in range(1000):
	random_probability = random.random()

	# check if we should explore or exploit
	if random_probability > epsilon: #EXPLOIT
		action_choice = bestAction(action_value_memory) # run the greedy algorithm on the action_value_memory map

		action_value_pair = np.array([[action_choice, reward(arms[action_choice])]]) # make an array representing a key value pair where the key is the index of this choice and the value is the reward this action confers

		# add the action and its reward to the memory map
		action_value_memory = np.concatenate((action_value_memory, action_value_pair), axis=0)
	else: # EXPLORE
		action_choice = np.where(arms == np.random.choice(arms))[0][0] # pick a random action index

		action_value_pair = np.array([[action_choice, reward(arms[action_choice])]]) # see what reward the random action confers

		# add the random action and its reward to the memory map
		action_value_pair = np.concatenate((action_value_memory, action_value_pair), axis=0)
	
	# calculate the percentage of the time that the correct arm is chosen
	# percCorrect = 100*(len(action_value_memory[np.where(action_value_memory[:,0] == np.argmax(arms))])/len(action_value_memory))

	#calculate the mean reward and plot it
	runningMean = np.mean(action_value_memory[:,1])
	plt.scatter(iteration, runningMean)

	if epsilon > 0:
		# apply some decay to epsilon
		epsilon = epsilon - iteration/700

plt.show()
sys.exit()