"""
OpenAI Gym implementation of platform domain, used for research of
Reinforcment Learning in parameterized action. See paper
"Reinforcement Learning with Parameterized Actions" by Warwick Masson,
Pravesh Ranchod and George Konidaris

Original code from https://github.com/WarwickMasson/aaai-platformer
"""
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from objects import (
    PlatformWorld,
    MAX_WIDTH,
    MAX_DX,
    ENEMY_SPEED,
    MAX_DX_ON_PLATFORM,
    GRAVITY,
    MAX_DDX,
    MAX_DY,
)


FPS = 100
VIEWER_WIDTH = int(MAX_WIDTH)
VIEWER_HEIGHT = 300

ACTION_LOOKUP = {
    0: 'run',
    1: 'hop',
    2: 'leap',
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

        # Determine the upperbound and lowerbound of action space

        # Bounds for run, see function accelerate and run,
        # -MAX_DDX < power / dt < MAX_DDX
        run_lower = np.array([-(MAX_DX - MAX_DX_ON_PLATFORM)])
        run_upper = np.array([MAX_DX - MAX_DX_ON_PLATFORM])

        # Bounds for hop and leap, see function accelerate, hop_to and leap_to
        # -MAX_DDX < diffx / time - self.velocity[0] < MAX_DY - dy0
        # Thus,
        # time * (-MAX_DDX) < diffx < time * (MAX_DY + MAX_DX - dy0)

        # Hop bounds
        hop_time = 2.0 * self.world.player.hop_dy0 / GRAVITY + 1.0
        hop_lower = np.array([hop_time * (-MAX_DDX)])
        hop_upper = np.array([hop_time * (MAX_DY + MAX_DX - self.world.player.hop_dy0)])
        # Leap bounds
        leap_time = 2.0 * self.world.player.leap_dy0 / GRAVITY + 1.0
        leap_lower = np.array([leap_time * (-MAX_DDX)])
        leap_upper = np.array([leap_time * (MAX_DY + MAX_DX - self.world.player.leap_dy0)])

        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(run_lower, run_upper),  # RUN
            spaces.Box(hop_lower, hop_upper),  # HOP
            spaces.Box(leap_lower, leap_upper)  # LEAP
        ))

        print(run_lower)
        print(run_upper)
        print(hop_lower)
        print(hop_upper)
        print(leap_lower)
        print(leap_upper)

        # The following feature boundary is getting from original code
        # using SHIFT_VECTOR, SCALE_VECTOR, and function scale_state.
        # Since the (state + SHIFT_VECTOR) / SCALE_VECTOR is in [0, 1]
        # We can derive that
        # 0 - SHIFT_VECTOR < state < SCALE_VECTOR - SHIFT_VECTOR,
        # Thus, we have the boundary as follow for action features

        # Also, note that in original learn.py code, the two policies,
        # for action determination and parameter determination using
        # different set of features (see function action_features and
        # parameter_features in FixedSarsaAgent), in here, we make them
        # a single feature vector.

        # Since the five additional features are all ratios in range [0, 1],
        # we set this as our observation space.
        high = np.array([
            MAX_WIDTH, MAX_DX, MAX_WIDTH, ENEMY_SPEED, 1.0, 1.0, 1.0, 1.0, 1.0
        ])

        low = np.array([
            -self.world.player.size[0], 0.0, 0.0, -ENEMY_SPEED, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
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
        # print('action_str is {}, action_param is {}'.format(action_str, action_param))
        state, reward, end_episode, step = self.world.take_action((action_str, action_param))
        return state, reward, end_episode, {step}

    def _reset(self):
        return self.world.reset()

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

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
