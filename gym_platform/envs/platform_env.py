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

from gym_platform.envs.objects import (
    Agent,
    Player,
    Platform,
    PLATFORMS_WIDTH,
    PLATFORM_HEIGHT,
    PLATFORMS_X,
    PLATFORMS_Y,
    GAPS,
    MAX_HEIGHT,
    MAX_WIDTH,
    MAX_PLATWIDTH,
    MAX_GAP,
    DT,
    MAX_DX,
    MAX_DY,
    MAX_DX_ON_PLATFORM,
    MAX_DDX,
    ENEMY_SPEED,
    GRAVITY,
)


FPS = 100
VIEWER_WIDTH = int(MAX_WIDTH)
VIEWER_HEIGHT = 300
ACTION_LOOKUP = {
    0: 'run',
    1: 'hop',
    2: 'leap',
    3: 'jump',
}


class PlatformEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'video.frame_per_second': FPS
    }

    def __init__(self):
        """
        Create the world and setup the action and observation spaces

        Parameters
        ----------
        call_render (bool): Due to the special structure of action (a action
            can last few step, and if we render, we should render every
            steps.), we need to call the render function inside our step
            function. Thus, we use this render flag to control this.
        """
        self._seed()
        self.viewer = None
        self.call_render = False
        self.mode = 'human'
        self.will_close = False

        self._create_world()
        self._reset()

        # Determine the upperbound and lowerbound of action space

        # Bounds for run, see function accelerate and run,
        # -MAX_DDX < dx_change / dt < MAX_DDX
        run_lower = np.array([-(MAX_DX - MAX_DX_ON_PLATFORM)])
        run_upper = np.array([MAX_DX - MAX_DX_ON_PLATFORM])

        # Bounds for hop and leap, see function accelerate, hop_to and leap_to
        # -MAX_DDX < diffx / time - self.velocity[0] < MAX_DY - dy0
        # Thus,
        # time * (-MAX_DDX) < diffx < time * (MAX_DY + MAX_DX - dy0)

        # Hop bounds
        hop_time = 2.0 * self.player.hop_dy0 / GRAVITY + 1.0
        hop_lower = np.array([hop_time * (-MAX_DDX)])
        hop_upper = np.array([hop_time * (MAX_DY + MAX_DX - self.player.hop_dy0)])
        # Leap bounds
        leap_time = 2.0 * self.player.leap_dy0 / GRAVITY + 1.0
        leap_lower = np.array([leap_time * (-MAX_DDX)])
        leap_upper = np.array([leap_time * (MAX_DY + MAX_DX - self.player.leap_dy0)])

        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(run_lower, run_upper),  # RUN
            spaces.Box(hop_lower, hop_upper),  # HOP
            spaces.Box(leap_lower, leap_upper)  # LEAP
        ))

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

        # high = np.array([
        #     MAX_WIDTH, MAX_DX, MAX_WIDTH, ENEMY_SPEED
        # ])

        low = np.array([
            -self.player.size[0], 0.0, 0.0, -ENEMY_SPEED, 0.0, 0.0, 0.0, 0.0, 0.0
        ])

        # low = np.array([
        #     -self.player.size[0], 0.0, 0.0, -ENEMY_SPEED
        # ])

        self.observation_space = spaces.Box(low, high)

    def _create_world(self):
        """
        The entities are setup and added to a space.
        """
        self.x_pos = 0.0
        self.player = Player(self.np_random)

        # Define platforms, we have three of them
        self.platforms = []
        for index in range(3):
            self.platforms.append(Platform(
                PLATFORMS_X[index], PLATFORMS_Y[index], PLATFORMS_WIDTH[index]))

        # Create enemies, we have two of them
        self.enemies = []
        for index in range(2):
            self.enemies.append(Agent(self.platforms[index], self.np_random))

        # Used to record the position of all agents
        self.states = []

    def platform_features(self, state):
        """
        This function returns the set of additional feature for
        determine policy parameters in original code.
        """
        xpos = state[0]
        if xpos < PLATFORMS_WIDTH[0] + GAPS[0]:
            pos = 0.0
            wd1 = PLATFORMS_WIDTH[0]
            wd2 = PLATFORMS_WIDTH[1]
            gap = GAPS[0]
            diff = PLATFORMS_Y[1] - PLATFORMS_Y[0]
        elif xpos < PLATFORMS_WIDTH[0] + GAPS[0] + PLATFORMS_WIDTH[1] + GAPS[1]:
            pos = PLATFORMS_WIDTH[0] + GAPS[0]
            wd1 = PLATFORMS_WIDTH[1]
            wd2 = PLATFORMS_WIDTH[2]
            gap = GAPS[1]
            diff = PLATFORMS_Y[2] - PLATFORMS_Y[1]
        else:
            pos = PLATFORMS_WIDTH[0] + GAPS[0] + PLATFORMS_WIDTH[1] + GAPS[1]
            wd1 = PLATFORMS_WIDTH[2]
            wd2 = 0.0
            gap = 0.0
            diff = 0.0
        return [wd1 / MAX_PLATWIDTH,
            wd2 / MAX_PLATWIDTH,
            gap / MAX_GAP,
            pos / MAX_WIDTH,
            diff / MAX_HEIGHT]

    def get_state(self):
        """
        Returns the representation of the current state.
        """
        if self.player.position[0] > self.platforms[1].position[0]:
            enemy = self.enemies[1]
        else:
            enemy = self.enemies[0]

        state = np.array([
            self.player.position[0],
            self.player.velocity[0],
            enemy.position[0],
            enemy.dx
        ])

        # Append the extra features
        extra_features = self.platform_features(state)
        state = np.append(state, extra_features)
        return state

    def on_platforms(self):
        """
        Checks if the player is on any of the platforms.
        """
        for platform in self.platforms:
            if self.player.on_platform(platform):
                return True
        return False

    def perform_action(self, action, dt=DT):
        """
        Applies for selected action for the given agent.
        """
        if self.on_platforms():
            if action:
                action, parameters = action
                if action == 'jump':
                    self.player.jump(parameters)
                elif action == 'run':
                    self.player.run(parameters, dt)
                elif action == 'leap':
                    self.player.leap_to(parameters)
                elif action == 'hop':
                    self.player.hop_to(parameters)
        else:
            self.player.fall()

    def lower_bound(self):
        """
        Returns the lowest height of the platforms.
        """
        length = []
        for platform in self.platforms:
            length.append(platform.position[1])
        lower = min(length)
        return lower

    def right_bound(self):
        """
        Returns the edge of the game.
        """
        return MAX_WIDTH

    def terminal_check(self, reward=0.0):
        """
        Determines if the episode is ended, and the reward.
        """
        # If we fall over the platform, episode over.
        done = self.player.position[1] < self.lower_bound() + PLATFORM_HEIGHT

        # If collide with enemy, episode over.
        for entity in self.enemies:
            if self.player.colliding(entity):
                done = True

        # If we go over the boundary, game over.
        right = self.player.position[0] >= self.right_bound()
        if right:
            reward = (self.right_bound() - self.x_pos) / self.right_bound()
            done = True
        return reward, done

    def update(self, action, dt=DT):
        """
        Performs a single transition with the given action,
        then returns the new state and a reward.
        """
        self.states.append([self.player.position.copy(),
                            self.enemies[0].position.copy(),
                            self.enemies[1].position.copy()])

        self.perform_action(action, dt)

        # Bound the velocity of player if is at floor
        if self.on_platforms():
            self.player.ground_bound()

        # Determine which enemy should move
        if self.player.position[0] > self.platforms[1].position[0]:
            enemy = self.enemies[1]
        else:
            enemy = self.enemies[0]

        # Update the position of agents
        for entity in [self.player, enemy]:
            entity.update(dt)

        # Test if we landed at any platform
        for i, platform in enumerate(self.platforms):
            if self.player.colliding(platform):
                self.player.decollide(platform)
                self.player.velocity[0] = 0.0

        # The reward for that step is the change in x value at that step
        reward = (self.player.position[0] - self.x_pos) / self.right_bound()
        return self.terminal_check(reward)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        """
        Take a full, stabilised update.

        Parameters
        ----------
        action (list): First is the discrete action, the rest is the parameters
            for all the discrete actions.
        """
        # TODO (ewei), action dispatcher seems stupid
        action_str = ACTION_LOOKUP[action[0]]
        action_param = action[action[0] + 1]

        # Record the position before the step for later reward computation
        self.x_pos = self.player.position[0]
        done = False
        step_unfinished = True
        action = (action_str, action_param)
        step = 0
        step_duration = 1.0
        # Keep updating until we finish the selected action
        while step_unfinished:
            if self.call_render:
                self.render_replacement()
            if action_str == 'run':
                reward, done = self.update(('run', abs(action_param)), DT)
                step_duration -= DT
                step_unfinished = step_duration > 0
            elif action_str in ['jump', 'hop', 'leap']:
                reward, done = self.update(action)
                step_unfinished = not self.on_platforms()
                # jump, hop, and leap only perform once, unlike run
                action = None
            if done:
                step_unfinished = False
            step += 1

        # Finish the loop, set the render flag to False
        self.call_render = False
        state = self.get_state()
        return state, reward, done, dict(step=step)

    def render(self, mode='human', close=False):
        """
        Due to the special structure of action, we need to call the
        render within the step function to have smoooth animation.

        Thus, to be compatible with the old way of calling render,
        we only set the flag for rendering here, also passes some
        arguments.
        """
        self.call_render = True
        self.mode = mode
        self.will_close = close

    def render_replacement(self):
        """
        This code copied from the core.py from gym. And we call
        this method inside the step function.
        """
        if not self.will_close: # then we have to check rendering mode
            modes = self.metadata.get('render.modes', [])
            if len(modes) == 0:
                raise error.UnsupportedMode('{} does not support rendering (requested mode: {})'.format(self, self.mode))
            elif self.mode not in modes:
                raise error.UnsupportedMode('Unsupported rendering mode: {}. (Supported modes for {}: {})'.format(self.mode, self, modes))
        return self._render(mode=self.mode, close=self.will_close)

    def _reset(self):
        """
        Reset the position of enemies and player, start
        a new episodes.
        """
        for enemy in self.enemies:
            enemy.reset()
        self.player.reset()
        self.x_pos = 0.0

        # Clear the buffer
        self.states = []
        return self.get_state()

    def _vertices(self, position, size):
        """
        Take the position and size and output the coordinates of the
        polygon.
        """
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
            for platform in self.platforms:
                position = platform.position
                size = platform.size
                vertices = self._vertices(position, size)
                polygon = rendering.make_polygon(v=vertices, filled=True)
                polygon.add_attr(rendering.Transform(translation=(0, baseline)))
                self.viewer.add_geom(polygon)

            # Define enemy movement data structure
            self.enemies_trans = []
            for enemy in self.enemies:
                self.enemies_trans.append(rendering.Transform())

            # Plot enemies
            for enemy, enemy_trans in zip(self.enemies, self.enemies_trans):
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
            position = self.player.start_position
            size = self.player.size
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
            position = self.enemies[index].position
            start_position = self.enemies[index].start_position
            tran.set_translation(position[0] - start_position[0],
                position[1] - start_position[1])

        # Move the player
        position = self.player.position
        start_position = self.player.start_position
        self.player_trans.set_translation(position[0] - start_position[0],
            position[1] - start_position[1])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
