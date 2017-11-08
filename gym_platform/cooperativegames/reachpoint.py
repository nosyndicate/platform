#""
#Classic cart-pole system implemented by Rich Sutton et al.
#Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
#"""

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
  #Use this if the values you'd pass in never 
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
		
		return height-y


	# Minimum toroidal difference between two values in the X dimension. 
	def tdx(self, x2, x1):
		width = self.width
		if abs(x1-x2) <= (width / 2):
			return x1 - x2
		
		dx = self.stx(x1) - self.stx(x2)
		if dx * 2 > width:
			return dx - width
		
		if dx * 2 < -width:
			return dx + width
		
		return dx



	def tdy(self, y2, y1):
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
class lion():
	def __init__(self, step_size, size, field, index):
		if(index ==0):
			self.x_pos = field.width/4
			self.y_pos = field.height/4
			self.theta = math.pi/4
		if(index == 1):
			self.x_pos = field.width/4
			self.y_pos = 3*field.height/4
			self.theta = -1*math.pi/4
		if(index == 2):
			self.x_pos = 3*field.width/4
			self.y_pos = 3*field.height/4
			self.theta = -3*math.pi/4
		if(index == 3):
			self.x_pos = 3*field.width/4
			self.y_pos = field.height/4
			self.theta = 3*math.pi/4
		
		self.step_size = 8;
		self.field = field;
		self.size = size;
		self.id = index		
	def do_step(self,delta,speed_percent):
		self.theta+=(delta-.5)#-self.step_size/2;

		sin_theta = math.sin(self.theta)
		cos_theta = math.cos(self.theta)
		x_change = cos_theta * self.step_size*speed_percent
		y_change = sin_theta * self.step_size*speed_percent
		#print("xpos is " + str(self.x_pos))
		self.x_pos = self.field.stx(self.x_pos+x_change);
		self.y_pos = self.field.sty(self.y_pos+y_change);

	def get_dist(self,lion):
		return math.sqrt((self.field.tdx(lion.x_pos,self.x_pos))*((self.field.tdx(lion.x_pos,self.x_pos))) + (self.field.tdy(self.y_pos,lion.y_pos))*(self.field.tdy(self.y_pos,lion.y_pos)))
	def colliding_with(self,lion):
		if self.id == lion.id:
			return False
		if(self.get_dist(lion) < self.size + lion.size):
				return True;
		return False;
class gazelle():
	
	def __init__(self, step_size, field, size, index):
		self.x_pos = field.width/2
		self.y_pos = field.height/2
		self.step_size = 9;
		self.field = field;
		self.size = size 
	def do_step(self,lions,index):
		
		
		vecX = 0;
		vecY = 0;
		
		count = 0;	
		for lion in lions:
			dist = self.get_dist(lion);
			#dist = dist+10
			field = self.field
			vecX += -1*self.field.tdx(self.x_pos,lion.x_pos)/(dist*dist) * (math.sqrt((field.width/2*field.width/2) * 2)-dist)   #/(dist*dist)*20
			vecY += -1*self.field.tdy(self.y_pos,lion.y_pos)/(dist*dist) * (math.sqrt((field.height/2*field.height/2) * 2)-dist)#/(dist*dist)*20	
	

		#	if(self.field.tdx(self.x_pos,lion.x_pos)>200 or True):
				
		#		print("count is: " + str(count))
		#		count = count+1;
		#		print("self: " + str(self.x_pos) + " " + str(self.y_pos) + " lion:" + str(lion.x_pos) + " " + str(lion.y_pos))
		#		print("dist is " + str(dist))
		#if(self.field.tdx(self.x_pos,lion.x_pos)>200 or True):
		#	print("vector is " + str(vecX) + " " + str(vecY))	
		if(vecX ==0):
			vecX=.01
		run_angle = math.atan(vecY/vecX)

		sin_theta = math.sin(run_angle)
		cos_theta = math.cos(run_angle)
		

		x_change = cos_theta * self.step_size
		y_change = sin_theta * self.step_size
		
		
		if(vecX>0):
			x_change = abs(x_change)
		else:
			x_change = -abs(x_change)
	
		if(vecY>0):
			y_change = abs(y_change)
		else:
			y_change = -abs(y_change)
		#if(self.field.tdx(self.x_pos,lion.x_pos)>200 or True):	
		#	print("final x change: " + str(x_change))
		#	print("final y change: " + str(y_change))
		
		x_change = 0;
		y_change = 0;
		self.x_pos = self.field.stx(self.x_pos+x_change);
		self.y_pos = self.field.sty(self.y_pos+y_change);


		#print("changed, xchange was " + str(x_change))
		#print("xpos is " + str(self.x_pos))
		#print()
	def get_dist(self,lion):
		return math.sqrt((self.field.tdx(lion.x_pos,self.x_pos))*((self.field.tdx(lion.x_pos,self.x_pos))) + (self.field.tdy(self.y_pos,lion.y_pos))*(self.field.tdy(self.y_pos,lion.y_pos)))
	def colliding_with(self,lion):
		if(self.get_dist(lion) < self.size + lion.size):
				return True;
		return False;
class ReachPointEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}
	def get_num_agents(self):
		return self.num_lions;
	def handleGazelleCollision(self):
		was_collision = False
		for lion in self.lions:
			for gazelle in self.gazellees:
				if(gazelle.colliding_with(lion)):
					return True
		return was_collision

	def handleLionCollision(self):

		total_collision = 0.0
		for lion in self.lions:
			for otherLion in self.lions:
				if(otherLion.colliding_with(lion)):
					total_collision = total_collision+1;
		return total_collision				
	
	def __init__(self,num_agents):
		self.viewer = None
		self.num_agents = num_agents
		self.field = ContinuousField(300,300)
		self.num_lions = 4;
		self.max_step = 1.0;
		self.lionSize =  50#10 too easy  # 10 too easy
		self.gazelleSize = 15 #25 too easy #12.5 too easy (not certain)
		self.lions = []
		for i in range(self.num_lions):
			self.lions.append(lion(self.max_step, self.lionSize, self.field, i))
		
		self.num_gazellees = 1;
		self.gazellees = []
		for i in range(self.num_gazellees):
			self.gazellees.append(gazelle(self.max_step, self.field, self.gazelleSize,i))
			
		
		self.lionCircles = [];
		self.lionTrans = [];
		self.gazelleTrans = [];
		self.gazellegazellees= [];

		# Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
		field = self.field

		self.action_space = spaces.Box(0,1, shape=(self.num_lions*2,))
		self.observation_space = spaces.Box(0,field.height, shape=(self.num_gazellees*2+self.num_lions*3,))

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
		reward  = -0.1;
		done = False;
		lionPositionArray=np.zeros((self.num_gazellees*2+self.num_lions*3));
		#print("size is " + str(self.num_lions+self.num_gazellees))
		count = 0;
		for i in range(self.num_lions):
			self.lions[i].do_step(action[i*2],action[i*2+1]);
			lionPositionArray[count] = self.field.tdx(self.gazellees[0].x_pos, self.lions[i].x_pos);
			count= count+1;
			lionPositionArray[count] = self.field.tdy(self.gazellees[0].y_pos,self.lions[i].y_pos);
			count= count+1;
			#lionPositionArray[count] = self.lions[i].theta
			vecX =  self.field.tdx(self.gazellees[0].x_pos, self.lions[i].x_pos);
			vecY = self.field.tdy(self.gazellees[0].y_pos,self.lions[i].y_pos);
			if(vecX==0):
				vecX = .0001;
			run_angle = math.atan(vecY/vecX)
			if(vecX <0):
				run_angle = run_angle-math.pi;
			if(abs(run_angle+math.pi*2)<run_angle):
				run_angle = run_angle+math.pi*2
			elif(abs(run_angle-math.pi*2)<run_angle):
				run_angle = run_angle-math.pi*2
			lionPositionArray[count] = run_angle;
			count= count+1;
			
			
		for i in range(self.num_gazellees):
			self.gazellees[i].do_step(self.lions,i);
			lionPositionArray[count] = 0
			count= count+1;
			lionPositionArray[count] = 0
			count = count+1;
		




		

		
		if(self.handleGazelleCollision()):
			# print("was good collision");
			# num_collided = self.handleGazelleCollision()
			# reward = 10.0/num_collided;
			# if(num_collided == 1):
			reward = 2

			done = True;
		
		
		if(self.handleLionCollision()>0):
			#print("Bad collidded with partner!!!!!!!!!!!!!!!!!!!!!!!");
			reward = -8
			
			
		
		min_dist = 10000;
		# for lion in self.lions:
		# 	dist = self.gazellees[0].get_dist(lion);
		# 	field = self.field
		# 	reward += (-1 * (dist - (lion.size + self.gazellees[0].size)) / (math.sqrt(self.field.width * self.field.width + self.field.height * self.field.height) - (lion.size + self.gazellees[0].size)) ) / self.num_lions 
			
		

		# if(done):
		# 	#print("achieved!")
		# 	reward = 2;
		
		return lionPositionArray, reward, done, {}

	def _reset(self):
		self.num_lions = 4;
		lionPositionArray=np.zeros((self.num_gazellees*2+self.num_lions*3));
		
		self.lions = []
		count = 0;
		self.num_gazellees = 1;
		self.gazellees = []
		
		for i in range(self.num_lions):
			self.lions.append(lion(self.max_step, self.lionSize, self.field, i))

		for i in range(self.num_gazellees):
			self.gazellees.append(gazelle(self.max_step, self.field,self.gazelleSize, i))
			lionPositionArray[count] = self.field.tdx(self.gazellees[0].x_pos, self.lions[i].x_pos);
			count= count+1;
			lionPositionArray[count] = self.field.tdy(self.gazellees[0].y_pos,self.lions[i].y_pos);
			count= count+1;
			#lionPositionArray[count] = self.lions[i].theta
			vecX =  self.field.tdx(self.gazellees[0].x_pos, self.lions[i].x_pos);
			vecY = self.field.tdy(self.gazellees[0].y_pos,self.lions[i].y_pos);
			if(vecX==0):
				vecX = .0001;
			run_angle = math.atan(vecY/vecX)
			if(vecX <0):
				run_angle = run_angle-math.pi;
			if(abs(run_angle+math.pi*2)<run_angle):
				run_angle = run_angle+math.pi*2
			elif(abs(run_angle-math.pi*2)<run_angle):
				run_angle = run_angle-math.pi*2
			lionPositionArray[count] = run_angle;


		lionPositionArray[count] = 0
		count= count+1;
		lionPositionArray[count] = 0
		count = count+1;
			
		
		
		self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.num_gazellees*2+self.num_lions*3,))

		self.steps_beyond_done = None
		return lionPositionArray

	def _render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		screen_width = self.field.width
		screen_height = self.field.height
		
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			
			for i in range(self.num_lions):
				self.lionCircles.append(rendering.make_circle(self.lions[i].size))
				print("xpos2 is " + str(self.lions[i].x_pos));
				self.lionTrans.append(rendering.Transform(translation=(self.lions[i].x_pos, self.lions[i].y_pos)))
				self.lionCircles[i].add_attr(self.lionTrans[i])
				self.lionCircles[i].set_color(.5,.5,.8)
				self.viewer.add_geom(self.lionCircles[i])
			for i in range(self.num_gazellees):
				self.gazellegazellees.append(rendering.make_circle(self.gazellees[i].size))
				self.gazellegazellees[i].set_color(.8,.6,.4)
				
				self.gazelleTrans.append(rendering.Transform(translation=(self.gazellees[i].x_pos,self.gazellees[i].y_pos )))
				self.gazellegazellees[i].add_attr(self.gazelleTrans[i])
				self.viewer.add_geom(self.gazellegazellees[i])
		if self.state is None: return None

		
		for i in range(self.num_lions):
			self.lionTrans[i].set_translation(self.lions[i].x_pos, self.lions[i].y_pos);
			#self.lionTrans[i].set_translation(self.field.tdx(self.gazellees[0].x_pos, self.lions[i].x_pos)+self.field.width/2, self.field.height+self.field.tdy( self.lions[i].y_pos,self.gazellees[0].y_pos));
		for i in range(self.num_gazellees):
			#print("xpos in print is " + str(self.gazellees[i].x_pos));
			self.gazelleTrans[i].set_translation(self.gazellees[i].x_pos, self.gazellees[i].y_pos);
			#self.gazelleTrans[i].set_translation(-self.field.width/2, -self.field.height/2);
			if( np.random.randint(0,10) >5):
				self.gazellegazellees[i].set_color(0,0,0)
			else:
				self.gazellegazellees[i].set_color(.8,.6,.4)

			if(self.handleGazelleCollision()):
				self.gazellegazellees[i].set_color(1,0,0)


		return self.viewer.render(return_rgb_array = mode=='rgb_array')