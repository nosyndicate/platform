

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)




class TwoLaneOneAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }


    def __init__(self):


        self.ballCircles = []
        self.ballTrans = []
        self.goal_threshold = 1.5
        self.width = 25
        self.height = 75
        self.max_dist = np.sqrt(self.width**2 + self.height**2)
        self.max_step = 1.;
        self.circleDiameter = 5.0
        self.circleRadius = self.circleDiameter / 2.0
        self.dest = [self.width / 2., 0]
        
        self.num_agents = 1
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]))
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.width, self.height]))

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def distToGoal(self):
        return np.sqrt((self.state[0] - self.dest[0]) ** 2 + (self.state[1] - self.dest[1]) ** 2)


    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # First two actions from agent 1, last two actions from agent 2
        self.state[0] = max(0.0 , min(self.state[0] + action[0], self.width))
        self.state[1] = max(0.0 , min(self.state[1] + action[1], self.height))
        action_cost = np.sum(action ** 2) * 0.1
        reward = 1 - self.distToGoal() / self.max_dist - action_cost
        done = self.distToGoal() < self.goal_threshold
        return self.state, reward, done, {}

    def _reset(self):
        self.state = np.array([self.width / 2., 7.])
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        scaleSize = 4
        screen_width = self.width * scaleSize
        screen_height = self.height * scaleSize

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            colors = [[.5,.5,.8], [.1,.1,.2]]

            for i in range(self.num_agents):
                self.ballCircles.append(rendering.make_circle(self.circleRadius * scaleSize))
                self.ballTrans.append(rendering.Transform(translation=(self.state[0] * scaleSize, self.state[1] * scaleSize)))
                self.ballCircles[i].add_attr(self.ballTrans[i])
                self.ballCircles[i].set_color(*colors[i])
                self.viewer.add_geom(self.ballCircles[i])

        if self.state is None: return None


        for i in range(self.num_agents):
            self.ballTrans[i].set_translation(self.state[0] * scaleSize, self.state[1] * scaleSize)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')