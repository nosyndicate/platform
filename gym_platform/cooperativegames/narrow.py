

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)
class ContinuousField(object):
	def __init__(self, width, height):
		self.width = width
		self.height = height

	# Simple [and fast] toroidal x.  Use this if the values you'd pass in never
	# stray beyond (-width ... width * 2) not inclusive.
	def stx(self, x):
		width = self.width
		if x >= 0:
			if x < width:
				return x
			return x - width
		return x + width


	# Simple [and fast] toroidal y.  Use this if the values you'd pass in never
	# stray beyond (-height ... height * 2) not inclusive.
	def sty(self, y):
		height = self.height
		if y >= 0:
			if y < height:
				return y
			return y - height

		return 0


	# Minimum toroidal difference between two values in the X dimension.
	def tdx(self, x1, x2):
		width = self.width
		if abs(x1-x2) <= (width / 2):
			return x1 - x2

		dx = self.stx(x1) - self.stx(x2)
		if dx * 2 > width:
			return dx - width

		if dx * 2 < -width:
			return dx + width

		return dx



	def tdy(self, y1, y2, height):
		height = self.height

		if abs(y1- y2) <= (height / 2):
			return y1 - y2

		dy = self.sty(y1) - self.sty(y2)
		if dy * 2 > height:
			return dy - height

		if dy * 2 < -height:
			return dy + height

		return dy

	# Minimum Toroidal difference vector between two points.
	# This subtracts the second point from the first and produces the minimum-length such subtractive vector,
	# considering wrap-around possibilities as well
	def tv(self, x1, x2, y1, y2):
		return self.tdx(x1, x2, self.width), self.tdy(y1, y2, self.height)

class Ball():
	def __init__(self, step_size, field):
		self.x_pos = 50.0
		self.y_pos = 50.0
		self.theta = math.pi/4;
		self.step_size = step_size;
		self.field = field;

	def do_step(self,delta):
		self.theta+=delta;
		sin_theta = math.sin(self.theta)
		cos_theta = math.cos(self.theta)
		x_change = cos_theta * self.step_size
		y_change = sin_theta * self.step_size
		#print("xpos is " + str(self.x_pos))
		self.x_pos = self.field.stx(self.x_pos+x_change);
		self.y_pos = self.field.sty(self.y_pos+y_change);
class Box():
	def __init__(self, step_size, field,index):
		self.x_pos = 100
		self.y_pos = 300+index*50
		self.step_size = step_size;
		self.field = field;
	def do_step(self,balls,index):
		self.theta+=delta;
		goalPos = 0.0
		if(index == 0):
			min_x = 0.0;
			for i in range(0,2):
				if(balls[i].x_pos < min_x):
					min_x = balls[i].x_pos;
			goalPos=min_x;
		else:
			max_x = 0.0;
			for i in range(0,2):
				if(balls[i].x_pos > max_x):
					max_x = balls[i].x_pos;
			goalPos=max_x;

		direction = 1;
		if(self.x_pos < goalPos):
			direction = -1;
		x_change = self.step_size*direction;


		self.x_pos = self.field.stx(self.x_pos+x_change);
		self.y_pos = self.field.sty(self.y_pos+y_change);
class NarrowEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}
	def get_num_agents(self):
		return 3

	def __init__(self, num_agents):
		self.field = ContinuousField(400,400)
		self.num_balls = 3;
		self.max_step = 10.0;
		self.balls = []
		for i in range(self.num_balls):
			self.balls.append(Ball(self.max_step, self.field))

		self.num_boxes = 2;
		self.boxes = []
		for i in range(self.num_boxes):
			self.boxes.append(Box(self.max_step, self.field,i))
		self.ballCircles = [];
		self.ballTrans = [];
		# Angle at which to fail the episode
		self.theta_threshold_radians = 12 * 2 * math.pi / 360
		self.x_threshold = 2.4

		# Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
		self.num_agents = num_agents

		self.action_space = spaces.Box(0,.5, shape=(self.num_balls,))
		self.observation_space = spaces.Box(0,400, shape=(self.num_balls+self.num_boxes,2))

		self._seed()
		self.viewer = None
		self.state = None

		self.steps_beyond_done = None

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = self.state
		reward  = -1;
		done = False;
		ballPositionArray=np.zeros((self.num_balls,2));
		for i in range(self.num_balls):
			self.balls[i].do_step(action[i]);
			ballPositionArray[i][0] = self.balls[i].x_pos;
			ballPositionArray[i][1] = self.balls[i].y_pos;
			if(ballPositionArray[i][1] > 360):
				done = True;
		if(done):
			reward = 1;

		return ballPositionArray, reward, done, {}

	def _reset(self):
		self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
		self.steps_beyond_done = None
		return np.array(self.state)

	def _render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		screen_width = 600
		screen_height = 400

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)

			for i in range(self.num_balls):
				self.ballCircles.append(rendering.make_circle(10))
				print("xpos2 is " + str(self.balls[i].x_pos));
				self.ballTrans.append(rendering.Transform(translation=(self.balls[i].x_pos, self.balls[i].y_pos)))
				self.ballCircles[i].add_attr(self.ballTrans[i])
				self.ballCircles[i].set_color(.5,.5,.8)
				self.viewer.add_geom(self.ballCircles[i])

		if self.state is None: return None


		for i in range(self.num_balls):
			self.ballTrans[i].set_translation(self.balls[i].x_pos, self.balls[i].y_pos);
		return self.viewer.render(return_rgb_array = mode=='rgb_array')