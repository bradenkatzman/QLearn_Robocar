# car_game.py using pygame and pymunk

import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

# initial variables to set up the pygame
width = 1000
height = 700

# initialize the game and set the screen size
pygame.init()
screen = pygame.display.set_mode((width, height))

clock = pygame.time.Clock() # we'll use this for time keeping

# alpha isn't used so it can be turned off
screen.set_alpha(None)

# if we show the sensors and redraw the screen, it will slow things down so we can better interpret the game
show_sensors = True
draw_screen = True

# variables that define the action IDs
LEFT = 0
RIGHT = 1

TURN_ANGLE = 0.2

# variables that affect the state of the environment
STATIC_OBSTACLE_UPDATE = 100 # frame frequency with which the (more) static obstacles should be moved
DYNAMIC_OBSTACLE_UPDATE = 5 # frame frequency with which the dynamic obstacles should be moved
VELOCITY = 20
SENSOR_SPREAD = 10 # the distance from the start to the end of the sensors
SENSOR_GAP = 20 # the gap before the first sensor
DYNAMIC_OBSTACLE_RADIUS = 15

class GameState:
	def __init__(self):
		self.crashed = False # the flag that will switch on if we come within a unit of an obstacle, triggering recovery

		# initialize a space and zero out its gravity, we're treating this as a 2d problem
		self.space = pymunk.Space()
		self.space.gravity = pymunk.Vec2d(0., 0.)

		# make a car
		self.create_car(100, 100, 0.5)

		# we want to keep a running total of the number of actions we take in a game to evaluate the model
		self.num_steps = 0

		# make walls to enclose the space
		static = [
			pymunk.Segment(
				self.space.static_body,
				(0, 1), (0, height), 1),
			pymunk.Segment(
				self.space.static_body,
				(1, height), (width, height), 1),
			pymunk.Segment(
				self.space.static_body,
				(width-1, height), (width-1, 1), 1),
			pymunk.Segment(
				self.space.static_body,
				(1, 1), (width, 1), 1)]

		for s in static:
			s.friction = 1
			s.group = 1
			s.collision_type = 1
			s.color = THECOLORS['red']
		self.space.add(static)

		# create pseudo-random obstacles
		self.static_obstacles = []
		self.static_obstacles.append(self.create_static_obstacle(200, 350, 75))
		self.static_obstacles.append(self.create_static_obstacle(700, 200, 50))
		self.static_obstacles.append(self.create_static_obstacle(600, 600, 35))

		# create a dynamic object
		self.dynamic_obstacles = []
		self.dynamic_obstacles.append(self.create_dynamic_obstacle())
		self.dynamic_obstacles.append(self.create_dynamic_obstacle())


#######################################################################
################ FUNCTIONS FOR DEFINING ###############################
################ THE GAME ENVIRONMENT #################################
#######################################################################

	# create a car at the starting position xy in the xy plane and give it a starting angle of r for which to face
	def create_car(self, x, y, r):
		# give the car some constant power so that it is always in motion
		inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))

		self.car_body = pymunk.Body(1, inertia)
		self.car_body.position = x,y

		self.car_shape = pymunk.Circle(self.car_body, 25)
		self.car_shape.color = THECOLORS["green"]
		self.car_shape.elasticity = 1.

		self.car_body.angle = r

		driving_direction = Vec2d(1,0).rotated(self.car_body.angle)

		# apply the driving direction to the car
		self.car_body.apply_impulse(driving_direction)

		self.space.add(self.car_body, self.car_shape)

	# create an object at x,y in the xy plane and give it a radius of r
	def create_static_obstacle(self, x, y, r):
		# set up the obstacle
		c_body = pymunk.Body(pymunk.inf, pymunk.inf)
		c_shape = pymunk.Circle(c_body, r)
		c_shape.elasticity = 1.
		c_body.position = x,y
		c_shape.color = THECOLORS["blue"]
		
		# add it to the environment
		self.space.add(c_body, c_shape)
		return c_body

	# create a dynamic object that will move in the environment and therefore help with overfitting - we'll just used a fixed location since if we make many, they'll move around quickly
	def create_dynamic_obstacle(self):
		# we need to give the object some intertia so that it will move
		inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
		dynamic_object_body = pymunk.Body(1, inertia)

		# give the object a starting position
		dynamic_object_body.position = 50, height - 100

		# make the object the shape of a circle and make its color pink
		dynamic_object_shape = pymunk.Circle(dynamic_object_body, DYNAMIC_OBSTACLE_RADIUS)
		dynamic_object_shape.color = THECOLORS["pink"]

		dynamic_object_shape.elasticity = 1.
		dynamic_object_shape.angle = 0.5

		# give the object a starting direction in which to move
		direction = Vec2d(1,0).rotated(dynamic_object_shape.angle)

		# add the object to the space
		self.space.add(dynamic_object_body, dynamic_object_shape)
		return dynamic_object_body

	def make_sonar_arm(self, x, y):
		arm_points = []

		# make an arm
		for i in range(1, 40):
			arm_points.append((SENSOR_GAP + x + (SENSOR_SPREAD * i), y))

		return arm_points


######################################################################
################ FUNCTIONS FOR PLAYING ###############################
################       THE GAME        ###############################
######################################################################

	# this function applies a given action at the current frame of the game
	def frame_step(self, action):
		# APPLY THE ACTION
		if action == LEFT:
			self.car_body.angle -= TURN_ANGLE
		elif action == RIGHT:
			self.car_body.angle += TURN_ANGLE

		# move the obstacles every OBSTACLE_UPDATE number of frames
		if self.num_steps % STATIC_OBSTACLE_UPDATE == 0:
			self.move_static_obstacles()

		if self.num_steps % DYNAMIC_OBSTACLE_UPDATE == 0:
			self.move_dynamic_obstacles()


		# move the car in the direction of the action by appling some velocity in that direction
		driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
		self.car_body.velocity = VELOCITY * driving_direction

		# make the necessary updates to the screen
		screen.fill(THECOLORS["black"])
		draw(screen, self.space) # update the screen
		self.space.step(1./10)
		if draw_screen:
			pygame.display.flip()
		clock.tick()

		# DETERMINE THE STATE collect the readings of the current location
		x,y = self.car_body.position

		sonar_readings = self.get_sonar_readings(x, y, self.car_body.angle)

		# normalize the readings so the numbers are cleaner to work with
		normalized_readings = [(x-20.)/20. for x in sonar_readings]

		# set the readings as the STATE of the game
		state = np.array([sonar_readings])

		# DETERMINE THE REWARD - we consider the car in a crash state if any of the sonar readings are 1 i.e. the car is within one unit of an obstacle
		if self.car_is_crashed(sonar_readings):
			self.crashed = True
			reward = -500
			self.recover_from_crash(driving_direction)
		else:
			# the higher the reading, the better the reward so we'll return the sum
			reward = -5 + int(self.sum_readings(sonar_readings) / 10)

		# increment the steps taken now that this frame has been fully processed
		self.num_steps += 1

		return reward, state

	# function to randomly move the (more) static obstacles around the environment
	def move_static_obstacles(self):
		for static_obstacle in self.static_obstacles:
			# choose a random speed value between 1 and 5 to apply to the object
			speed = random.randint(1, 3)

			# choose a random direction to move the object in, between -2 and 2
			direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))

			if random.randint(0, 1) >= 0.5:
				static_obstacle.velocity = speed * direction
			else:
				static_obstacle.velocity = speed * -direction

	def move_dynamic_obstacles(self):
		for dynamic_obstacle in self.dynamic_obstacles:
			speed = random.randint(10, 50)

			direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-1, 1))

			if random.randint(0, 1) >= 0.5:
				dynamic_obstacle.velocity = speed * direction
			else:
				dynamic_obstacle.velocity = speed * -direction


	# given the current readings, determine if the car crashed
	def car_is_crashed(self, readings):
		if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
			return True
		else:
			return False

	# the car entered a crash state, so we want to back up at a slight angle until we're not in a crash state
	def recover_from_crash(self, driving_direction):
		while self.crashed:
			# back up at an angle
			self.car_body.velocity = -VELOCITY * driving_direction
			self.crashed = False

			for i in range(10):
				# turn slightly
				self.car_body.angle += TURN_ANGLE

				# update the screen to reflect the recovery changes
				screen.fill(THECOLORS["red"])
				draw(screen, self.space)
				self.space.step(1./10)
				if draw_screen:
					pygame.display.flip()
				clock.tick()

	# take a sum of the readings to use as the reward
	def sum_readings(self, readings):
		total = 0
		for reading in readings:
			total += reading
		return total

	# Return distance readings from each sensors. The distance is defined as a count of the first non-zero reading starting at the sensors
	def get_sonar_readings(self, x, y, angle):
		readings = []

		# make 3 sonar arms
		arm_left = self.make_sonar_arm(x, y)
		arm_middle = arm_left
		arm_right = arm_left

		# rotate the arms so that they create a fan of readings in front of the car, and query their readings
		readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75)) # 45 degrees left
		readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0)) # straight down the middle
		readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75)) # 45 degrees right

		if show_sensors:
			pygame.display.update()

		return readings

	def get_arm_distance(self, arm, x, y, angle, offset):
		distance_counter = 0

		# check at each point on each iteration and see if we've hit something
		for point in arm:
			distance_counter += 1

			# move the point to the right spot
			rotated_p = self.get_rotated_point(x, y, point[0], point[1], angle + offset)

			# check if we hit something (either an object or a wall) and return the current distance if we have
			if rotated_p[0] <= 0 or rotated_p[1] <= 0 or rotated_p[0] >= width or rotated_p[1] >= height:
				return distance_counter
			else:
				obstacle = screen.get_at(rotated_p)
				if self.get_track_or_not(obstacle) != 0:
					return distance_counter

			if show_sensors:
				pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

		return distance_counter


	def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
		# rotate x_2 and y_2 around x_1 and y_1 by the angle given by radians
		x_change = ((x_2 - x_1) * math.cos(radians)) + ((y_2 - y_1) * math.sin(radians))
		y_change = ((y_1 - y_2) * math.cos(radians)) - ((x_1 - x_2) * math.sin(radians))

		new_x = x_change + x_1
		new_y = height - (y_change + y_1)
		return int(new_x), int(new_y)

	def get_track_or_not(self, readings):
		if readings == THECOLORS["black"]:
			return 0
		else:
			return 1


if __name__ == "__main__":
	game_state = GameState()
	while True:
		game_state.frame_step((random.randint(0, 2)))