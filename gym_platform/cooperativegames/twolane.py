

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
    def __init__(self, step_size, field, initX, initY, size, id):
        self.x_pos = initX
        self.y_pos = initY
        self.theta = math.pi/4
        self.step_size = step_size
        self.field = field
        self.max_dist = np.sqrt(field.width ** 2 + field.height ** 2)
        self.size = size
        self.radius = size / 2.0
        self.id = id

    def setDestPoint(self, dest):
        self.dest = dest

    def getPoints(self, prevX, prevY):
        return [[MyPoint(self.x_pos, self.y_pos), MyPoint(prevX, prevY)], [MyPoint(self.x_pos, self.y_pos + self.radius), MyPoint(prevX, prevY + self.radius)], [MyPoint(self.x_pos, self.y_pos - self.radius), MyPoint(prevX, prevY - self.radius)], [MyPoint(self.x_pos + self.radius, self.y_pos), MyPoint(prevX + self.radius, prevY)], [MyPoint(self.x_pos - self.radius, self.y_pos), MyPoint(prevX - self.radius, prevY)]]

    def do_step(self,xvel, yvel):
        x_change = xvel * self.step_size
        y_change = yvel * self.step_size
        #print("xpos is " + str(self.x_pos))
        self.x_pos = self.field.sx(self.x_pos+x_change)
        self.y_pos = self.field.sy(self.y_pos+y_change)
        #print("when moved i am at ({}, {})".format(self.x_pos, self.y_pos))
    def get_dist(self,ball):
        return math.sqrt((self.field.dx(ball.x_pos,self.x_pos))*((self.field.dx(ball.x_pos,self.x_pos))) + (self.field.dy(self.y_pos,ball.y_pos))*(self.field.dy(self.y_pos,ball.y_pos)))
    def colliding_with(self,ball):
        if self.id ==  ball.id:
            return False
        if(self.get_dist(ball) < self.radius + ball.radius):
            return True
        return False

    # def currentReward(self):
    #     return 1. / math.sqrt(math.pow((self.x_pos - self.dest.x),2) + math.pow((self.y_pos - self.dest.y),2))

    def currentReward(self):
        dist = np.sqrt((self.x_pos - self.dest.x) ** 2 + (self.y_pos - self.dest.y) ** 2)
        return dist / self.max_dist

class MyPoint():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Box():
    def __init__(self, field, xpos_left, ypos_left, xpos_right, ypos_right, height):

        ## these define the line that is the bottom of the rectangle
        ## then height is how high the rectange is so translate the y coordinates +height
        self.x_pos_left = xpos_left
        self.x_pos_right = xpos_right
        self.y_pos_left = ypos_left
        self.y_pos_right = ypos_right
        self.size = height
        #[(l,b), (l,t), (r,t), (r,b)]
        #self.rect = [(self.x_pos_left, self.y_pos_left), (self.x_pos_left, self.y_pos_left + self.size), (self.x_pos_right, self.y_pos_right+self.size), (self.x_pos_right, self.y_pos_right)]
        self.rect = [[self.x_pos_left, self.y_pos_left], [self.x_pos_left, self.y_pos_left + self.size], [self.x_pos_right, self.y_pos_right+self.size], [self.x_pos_right, self.y_pos_right]]
        print("({}, {}) ({}, {})".format(self.x_pos_left, self.y_pos_left, self.x_pos_right, self.y_pos_right))
        self.bottomLeft = MyPoint(self.x_pos_left, self.y_pos_left)
        self.topLeft = MyPoint(self.x_pos_left, self.y_pos_left + self.size)
        self.topRight = MyPoint(self.x_pos_right, self.y_pos_right+self.size)
        self.bottomRight = MyPoint(self.x_pos_right, self.y_pos_right)

        self.sides = [[self.bottomLeft, self.bottomRight], [self.bottomRight, self.topRight], [self.topLeft, self.topRight], [self.bottomLeft, self.topLeft]]

        self.field = field
    def ccw(self, A,B,C):
        return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

    # Return true if line segments AB and CD intersect
    def intersect(self,A,B,C,D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

    def collideWith(self, ball, prevX, prevY):
        circleLines = ball.getPoints(prevX, prevY)
        for boxSide in self.sides:
            for circleLine in circleLines:
                if self.intersect(boxSide[0], boxSide[1], circleLine[0], circleLine[1]):
                    return True

class TwoLaneEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def get_num_agents(self):
        return 2

    def __init__(self, sameSide):

        print("same side = {}".format(sameSide))
        self.sameSide = sameSide

        self.ballCircles = []
        self.ballTrans = []
        self.boxRects = []
        self.boxTrans = []

        self.field = ContinuousField(50, 75)
        self.num_balls = self.get_num_agents();
        self.max_step = 1.;
        self.circleDiameter = 5.0
        self.circleRadius = self.circleDiameter / 2.0
        self.BoxHeight = self.circleDiameter * 10
        self.passageWidth = self.circleDiameter * 2.5
        self.reset_state()
        self.num_boxes = 2;
        self.boxes = []
        # left box (0, height/2- self.BoxHeight / 2.) (width / 2 - passageWidth / 2, height/2 - self.BoxHeight / 2.)
        self.boxes.append(
            Box(self.field, 0,
                self.field.height / 2. - self.BoxHeight / 2.,
                self.field.width / 2. - self.passageWidth / 2.0,
                self.field.height / 2. - self.BoxHeight / 2.,
                self.BoxHeight))
        # right box (width / 2 + passageWidth / 2, height/2 - self.BoxHeight / 2.) (width, height/2- self.BoxHeight / 2.)
        self.boxes.append(
            Box(self.field, self.field.width / 2. + self.passageWidth / 2.0,
                self.field.height / 2. - self.BoxHeight / 2.,
                self.field.width, self.field.height / 2. - self.BoxHeight / 2.,
                self.BoxHeight))

        self.num_agents = self.get_num_agents()

        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]))
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([self.field.width, self.field.height,
                self.field.width, self.field.height]))

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # First two actions from agent 1, last two actions from agent 2
        action = action.reshape(-1, 2)
        state = self.state
        reward  = -1;
        done = False;
        ballPositionArray=np.zeros((self.num_balls,4));
        for i in range(self.num_balls):
            prevx = self.balls[i].x_pos
            prevy = self.balls[i].y_pos
            #print("action {} prev ({}, {}) id = {}".format(action[i], prevx, prevy, self.balls[i].id))
            self.balls[i].do_step(action[i][0], action[i][1])
            #print("New location = ({}, {}) id = {}".format(self.balls[i].x_pos, self.balls[i].y_pos, self.balls[i].id))
            collide = False
            for j in range(self.num_balls):
                if self.balls[i].colliding_with(self.balls[j]):
                    collide = True
            if collide:
                print("collided")
            else:
                ## now check if the agent collided with the boxes
                boxCollide = False
                for j in range(len(self.boxes)):
                    #continue
                    #print("New location = ({}, {}) id = {}".format(self.balls[i].x_pos, self.balls[i].y_pos, self.balls[i].id))

                    if (self.boxes[j].collideWith(self.balls[i],prevx, prevy)):
                        ## then we have intersected with a box so reset
                        #print("Collided with box")
                        #print("prev ({}, {}) new ({} {}) id {}".format(prevx, prevy, self.balls[i].x_pos,self.balls[i].y_pos, self.balls[i].id))
                        self.balls[i].x_pos = prevx
                        self.balls[i].y_pos = prevy
                        ballPositionArray[i][0] = prevx
                        ballPositionArray[i][1] = prevy
                        boxCollide = True

                if boxCollide == False:
                    ballPositionArray[i][0] = self.balls[i].x_pos;
                    ballPositionArray[i][1] = self.balls[i].y_pos;

        if


        ## now compute the reward
        if self.sameSide:
            ## not sure yet...
            reward = 0
        else:
            reward = 0.
            reachedSideReward = 5

            # if self.balls[0].y_pos > self.boxes[0].topLeft.y:
            #     reward = reward + reachedSideReward
            # else:
            reward = reward + self.balls[0].currentReward()

            # if self.balls[1].y_pos < self.boxes[0].bottomLeft.y:
            #     reward = reward + reachedSideReward
            # else:
            reward = reward + self.balls[1].currentReward()

        return self.get_state(self.balls[0], self.balls[1]), reward, done, {}


    def get_state(self, ball1, ball2):
        # [[my_x, my_y, rel_x, rel_y],[my_x, my_y, rel_x, rel_y]]
        return np.array([[ball1.x_pos, ball1.y_pos, ball2.x_pos - ball1.x_pos, ball2.y_pos - ball1.y_pos],
        [ball2.x_pos, ball2.y_pos, ball1.x_pos - ball2.x_pos, ball1.y_pos - ball2.y_pos]])

    def reset_state(self):
        self.balls = []
        ## we are on the same side of the passage and we need to make it to the otherside
        # bottom left ball
        self.balls.append(Ball(self.max_step, self.field, self.field.width / 2.,  self.circleDiameter, self.circleDiameter, 0))
        self.balls[0].setDestPoint(MyPoint(self.field.width / 2., self.field.height))
        # top right ball
        self.balls.append(Ball(self.max_step, self.field, self.field.width/ 2., self.field.height - self.circleDiameter, self.circleDiameter, 1))
        self.balls[1].setDestPoint(MyPoint(self.field.width / 2., 0))
        self.state = self.get_state(self.balls[0], self.balls[1])


    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.num_agents = 2
        self.reset_state()
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        scaleSize = 4
        screen_width = self.field.width * scaleSize
        screen_height = self.field.height * scaleSize

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            colors = [[.5,.5,.8], [.1,.1,.2]]

            for i in range(self.num_balls):
                self.ballCircles.append(rendering.make_circle(self.circleRadius * scaleSize))
                self.ballTrans.append(rendering.Transform(translation=(self.balls[i].x_pos * scaleSize, self.balls[i].y_pos * scaleSize)))
                self.ballCircles[i].add_attr(self.ballTrans[i])
                # self.ballCircles[i].set_color(.5,.5,.8)
                self.ballCircles[i].set_color(*colors[i])
                self.viewer.add_geom(self.ballCircles[i])
            for i in range(self.num_boxes):
                myscalerect = self.boxes[i].rect
                for a in myscalerect:
                    a[0] = a[0] * scaleSize
                    a[1] = a[1] * scaleSize
                self.boxRects.append(rendering.make_polygon(myscalerect))
                #self.boxRects.append(rendering.make_polygon(self.boxes[i].rect))
                #self.boxTrans.append(rendering.Transform(translation=(self.balls[i].x_pos, self.balls[i].y_pos)))
                #self.boxRects[i].add_attr(self.boxTrans[i])
                self.boxRects[i].set_color(.5,.5,.8)
                self.viewer.add_geom(self.boxRects[i])

        if self.state is None: return None


        for i in range(self.num_balls):
            #print("ball [{}, {}]".format(self.balls[i].x_pos, self.balls[i].y_pos))
            self.ballTrans[i].set_translation(self.balls[i].x_pos * scaleSize, self.balls[i].y_pos * scaleSize)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')