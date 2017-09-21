import logging
from os import path

import gym
from gym import error, spaces, utils
from gym.envs.classic_control import rendering
from gym.utils import seeding
import numpy as np


from objects import (
    PlatformWorld,
    MAX_WIDTH
)


FPS = 50
VIEWER_WIDTH = int(MAX_WIDTH)
VIEWER_HEIGHT = 300

ACTION_LOOKUP = {
    0 : 'run',
    1 : 'hop',
    2 : 'leap',
}

class PlatformEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'video.frame_per_second': FPS
    }

    def __init__(self):
        ''' The entities are set up and added to a space. '''
        self._seed()
        self.viewer = None

        self.world = PlatformWorld()
        self._reset()

        # each action should have a different space of parameter
        # TODO (ewei), this is a fake action space for now, need to adjust.
        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(np.array([-1]), np.array([+1])),
            spaces.Box(np.array([-1]), np.array([+1])),
            spaces.Box(np.array([-1]), np.array([+1]))
        ))

        # TODO (ewei) original code have implicit state
        # need to figure where are they, in addition, the feature using
        # MAX_WIDTH is no accurate, need to figure out what's the real
        # value, or is the inaccurate value harmless.
        high = np.array([MAX_WIDTH, 100.0, MAX_WIDTH, 30.0])
        low = np.array([0.0, 0.0, 0.0, -30.0])
        self.observation_space = spaces.Box(low, high)


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        """
        Parameters
        ----------
        action (list): First is the discrete action, the rest is the parameters
            for all the discrete actions.
        """
        # TODO (ewei), action dispatcher seems stupid
        action_str = ACTION_LOOKUP[action[0]]
        action_param = action[action[0] + 1]
        state, reward, end_episode, step = self.world.take_action((action_str, action_param))
        return state, reward, end_episode, {step}

    def _reset(self):
        self.world.reset()


    def _vertices(self, position, size):
        left, right, top, bottom = \
            position[0], position[0] + size[0], position[1], position[1] + size[1]
        v = [(left, bottom), (left, top), (right, top), (right, bottom)]
        return v

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            # Base height for the platforms
            baseline = 10

            self.viewer = rendering.Viewer(VIEWER_WIDTH, VIEWER_HEIGHT)
            # Plot platforms
            for platform in self.world.platforms:
                position = platform.position
                size = platform.size
                vertices = self._vertices(position, size)
                polygon = rendering.make_polygon(v=vertices, filled=True)
                polygon.add_attr(rendering.Transform(translation=(0, baseline)))
                self.viewer.add_geom(polygon)

            # Define enemy movement data structure
            self.enemies_trans = []
            for enemy in self.world.enemies:
                self.enemies_trans.append(rendering.Transform())

            # Plot enemies
            for enemy, enemy_trans in zip(self.world.enemies, self.enemies_trans):
                position = enemy.start_position
                size = enemy.size
                vertices = self._vertices(position, size)
                enemy_polygon = rendering.make_polygon(v=vertices, filled=True)
                enemy_polygon.add_attr(enemy_trans)
                # Set enemy to crimson
                enemy_polygon.set_color(0.862745, 0.0784314, 0.235294)
                enemy_polygon.add_attr(rendering.Transform(translation=(0, baseline)))
                self.viewer.add_geom(enemy_polygon)

            # Plot player
            position = self.world.player.start_position
            size = self.world.player.size
            vertices = self._vertices(position, size)
            player_polygon = rendering.make_polygon(v=vertices, filled=True)
            self.player_trans = rendering.Transform()
            player_polygon.add_attr(self.player_trans)
            # Set player to deepskyblue
            player_polygon.set_color(0, 0.74902, 1)
            player_polygon.add_attr(rendering.Transform(translation=(0, baseline)))
            self.viewer.add_geom(player_polygon)

        # Move the enemies
        for index, tran in enumerate(self.enemies_trans):
            position = self.world.enemies[index].position
            start_position = self.world.enemies[index].start_position
            tran.set_translation(position[0] - start_position[0],
                position[1] - start_position[1])

        # Move the player
        position = self.world.player.position
        start_position = self.world.player.start_position
        self.player_trans.set_translation(position[0] - start_position[0],
            position[1] - start_position[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')



