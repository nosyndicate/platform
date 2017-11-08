

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


	def sx(self, x):
		if x > self.width:
			return self.width
		elif x < 0:
			return 0
		else:
			return x
	def sy(self, y):
		if y > self.height:
			return self.height
		elif y < 0:
			return 0
		else:
			return y
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

	def dx(self, x1, x2):
		return x1 - x2

	def dy(self, y1, y2):
		return y1 - y2

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
	def __init__(self, step_size, field, initX, initY, size):
		self.x_pos = initX
		self.y_pos = initY
		self.theta = math.pi/4;
		self.step_size = step_size;
		self.field = field;
		self.size = size
	def do_step(self,xvel, yvel):
		
		x_change = xvel * self.step_size
		y_change = yvel * self.step_size
		#print("xpos is " + str(self.x_pos))


		self.x_pos = self.field.sx(self.x_pos+x_change);
		self.y_pos = self.field.sy(self.y_pos+y_change);
	def get_dist(self,ball):
		return math.sqrt((self.field.tx(ball.x_pos,self.x_pos))*((self.field.tx(ball.x_pos,self.x_pos))) + (self.field.ty(self.y_pos,ball.y_pos))*(self.field.ty(self.y_pos,ball.y_pos)))
	def colliding_with(self,ball):
		if self.id ==  ball.id:
			return False
		if(self.get_dist(ball) < self.size + ball.size):
				return True;
		return False;


class Box():
	def __init__(self, field, xpos_left, ypos_left, xpos_right, ypos_right, height):

		## these define the line that is the bottom of the rectangle
		## then height is how high the rectange is so translate the y coordinates +height
		self.x_pos_left = xpos_left
		self.x_pos_right = xpos_right
		self.y_pos_left = ypos_left
		self.y_pos_right = ypos_right
		self.size = height
		self.field = field
	
	def lineCollides(self, startx, starty, endx, endy):
		#Output
		# True if lines collide
		denom = ((endy – starty) * (self.x_pos_right – self.x_pos_left)) –
		((endx – startx) * (self.y_pos_right - self.y_pos_left))
		return denom != 0

class NarrowEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}
	def get_num_agents(self):
		return 2

	def __init__(self, num_agents):
		self.field = ContinuousField(400,400)
		self.num_balls = num_agents;
		self.max_step = 10.0;
		self.circleDiameter = 5.0
		self.circleRadius = self.circleDiameter / 2.0
		self.BoxHeight = self.circleDiameter
		self.balls = []
		for i in range(self.num_balls):
			self.balls.append(Ball(self.max_step, self.field, 3, 4, self.circleDiameter))

		self.num_boxes = 2;
		self.boxes = []
		self.boxes.append(Box(self.field, 0, self.field.height / 2., self.field.width / 2. - self.circleRadius * 1.5, self.field.height / 2., self.BoxHeight))
		self.boxes.append(Box(self.field, self.field.width / 2. + self.circleRadius * 1.5, self.field.height / 2., self.field.width, self.field.height / 2., self.BoxHeight))
		
		
		self.num_agents = num_agents

		self.action_space = spaces.Box(low=-1.,high=1., shape=(self.num_balls,2))
		self.observation_space = spaces.Box(0,400, shape=(self.num_balls,2))

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
			prevx = self.balls[i].x_pos
			prevy = self.balls[i].y_pos
			self.balls[i].do_step(action[i])
			collide = False
			for i in range(self.num_balls):
				if self.balls[i].colliding_with(self.balls[i]):
					collide = True
			if collide:

			else:
				## now check if the agent collided with the boxes
				boxCollide = False
				for i in range(len(self.boxes)):
					if (self.boxes[i].lineCollides(prevx, prevy, self.balls[i].x_pos,self.balls[i].y_pos) or
						self.boxes[i].lineCollides(prevx - self.circleRadius, prevy, self.balls[i].x_pos - self.circleRadius, self.balls[i].y_pos) or
						self.boxes[i].lineCollides(prevx + self.circleRadius, prevy, self.balls[i].x_pos + self.circleRadius, self.balls[i].y_pos)):
						## then we have intersected with a box so reset
						self.balls[i].x_pos = prevx
						self.balls[i].y_pos = prevy
						ballPositionArray[i][0] = prevx
						ballPositionArray[i][1] = prevy
						boxCollide = True
				if boxCollide == false:
					ballPositionArray[i][0] = self.balls[i].x_pos;
					ballPositionArray[i][1] = self.balls[i].y_pos;

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