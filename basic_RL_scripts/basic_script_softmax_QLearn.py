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

action_value_memory = np.ones(n) # just an array of ones because we will store running reward means to be more energy conscious

counts = np.zeros(n) # keep a lit of how many times we've taken each action to use to calculate the average reward over time

action_value_softmax_probabilities = np.zeros(n) # an array of zeros that will store the softmax generated probability ranks for each action
action_value_softmax_probabilities[:] = 0.1 # initialize all softmax probability to 0.1 so that they start at the same place

# each arm has a probability. we'll set the reward for each arm by looping to 10, and at each step 
# doing the following algorithm
def reward(prob):
	reward = 0
	for i in range(10):
		if random.random() < prob: # if a random number [0,1] is less than the random probability assigned to this arm, incrememnt the reward value
			reward += 1
	return reward # note that the maximum reward is 10

# softmax algorithm for selecting the action with the highest weighted probability
tau = 0.5
def softmax(action_value_memory):
	# initialize empty array for the probabilities of n actions
	probabilities = np.zeros(n)

	for i in range(n):
		# compute the softmax equation by taking the exponential of this action value over tau and dividing it by the sum of action values over tau
		softmax_value = (np.exp(action_value_memory[i] / tau)) / np.sum(np.exp(action_value_memory[:] / tau))

		# set the softmax value as the probability of this action
		probabilities[i] = softmax_value

	return probabilities



###############################################################################################
# main play loop
fig = plt.figure()
plt.xlabel("Plays")
plt.ylabel("Avg Reward")
for iteration in range(500):
	# make a "random" selection of an action using -- np.random.choice generates a random sample from the given array
	random_softmax_probabilities_choice = np.where(arms == np.random.choice(arms, p=action_value_softmax_probabilities))[0][0]

	# increment the counter for this action in the counts array
	counts[random_softmax_probabilities_choice] += 1

	# now grab that incremented value to be used to update the average reward for this action after we calculate the reward we get now
	action_count = counts[random_softmax_probabilities_choice]

	# calculate the reward for this action and update the running average for this action
	rwd = reward(arms[random_softmax_probabilities_choice])
	old_avg = action_value_memory[random_softmax_probabilities_choice]
	new_avg = old_avg + (rwd - old_avg)/action_count

	# put this updated average into the memory map
	action_value_memory[random_softmax_probabilities_choice] = new_avg

	# update the softmax values for the next play
	action_value_softmax_probabilities = softmax(action_value_memory)

	# keep a tab of the running mean and add it to the scatter plot
	runningMean = np.average(action_value_memory, weights=np.array([counts[j]/np.sum(counts) for j in range(len(counts))]))
	plt.scatter(iteration, runningMean)

plt.show()
sys.exit()