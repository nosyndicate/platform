import numpy as np
from numpy.random import uniform

# Parameters for the platform
PLATFORMS_WIDTH = [250, 275, 50]
PLATFORM_HEIGHT = 40
PLATFORMS_X = [0.0, 475, 985]
PLATFORMS_Y = [0.0, 0.0, 0.0]
# Width of the gaps
GAPS = [225, 235]

MAX_HEIGHT = max(1.0, max(PLATFORMS_Y))
MAX_PLATWIDTH = max(PLATFORMS_WIDTH)
MAX_WIDTH = sum(PLATFORMS_WIDTH + GAPS)
MAX_GAP = max(GAPS)

# Duration of a time unit, some action may last several time unit
DT = 0.05

MAX_DX = 100.0
MAX_DY = 200.0
# Max x-axis velocity for player on the platform
MAX_DX_ON_PLATFORM = 70.0
# Max x-axis acceleration
MAX_DDX = (MAX_DX - MAX_DX_ON_PLATFORM) / DT
# Max y-axis acceleration
MAX_DDY = MAX_DY / DT
# Agent movement speed
ENEMY_SPEED = 30.0
LEAP_DEV = 1.0
HOP_DEV = 1.0
ENEMY_NOISE = 0.5
CHECK_SCALE = False
GRAVITY = 9.8




def point(x, y):
    """
    Return a numpy array represent a point in 2D surface.
    """
    return np.array([float(x), float(y)])


def vector(x, y):
    """
    Return a numpy array represent a vector in 2D surface
    """
    return point(x, y)


def bound(value, lower_b, upper_b):
    """
    Clips off a value which exceeds the lower or upper bounds.
    """
    if value < lower_b:
        return lower_b
    elif value > upper_b:
        return upper_b
    else:
        return value


def bound_vector(value, x_max, y_max):
    """
    Bounds a vector between a negative and positive maximum range.
    """
    xval = bound(value[0], -x_max, x_max)
    yval = bound(value[1], -y_max, y_max)
    return vector(xval, yval)


class Platform(object):
    """
    Represent a fixed platform.
    """
    def __init__(self, x, y, width):
        self.position = point(x, y)
        self.size = vector(width, PLATFORM_HEIGHT)


class Agent(object):
    """
    Defines the enemy.
    """
    size = vector(20.0, 30.0)

    def __init__(self, platform):
        """
        Initialize the enemy on the platform.
        """
        self.platform = platform

        # Define properties of agent.
        # dx is the velocity of enemy on x-axis
        self.dx = -ENEMY_SPEED
        # Set the the position of agent to the end of platform
        self.position = self.platform.size + self.platform.position
        # Due to size of agent, need to move it a little inside
        self.position[0] -= self.size[0]

        # Set the leftmost and rightmost position the agent can be
        self.leftmost = self.platform.position[0]
        self.rightmost = self.position[0]

        # To identify the start position, used for rendering
        self.start_position = self.position.copy()

    def reset(self):
        """
        Reset the location of agent.
        """
        self.dx = -ENEMY_SPEED
        self.position = self.platform.size + self.platform.position
        self.position[0] -= self.size[0]

    def update(self, dt):
        """
        Shift the enemy along the platform.

        Parameters
        ----------
        dt (): time interval of movement.
        """
        # If go out of the range, turn around
        if not self.leftmost < self.position[0] < self.rightmost:
            self.dx *= -1
        # Add some noise to agent's action, e.g. velocity on x-axis
        self.dx += np.random.normal(0.0, ENEMY_NOISE * dt)
        # Make sure the speed is in valid range.
        self.dx = bound(self.dx, -ENEMY_SPEED, ENEMY_SPEED)
        # Update the position of agent
        self.position[0] += self.dx * dt
        # Make sure the position is in valid range
        self.position[0] = bound(self.position[0], self.leftmost, self.rightmost)


class Player(Agent):
    """
    Represent the player character.
    """
    velocity_decay = 0.99

    def __init__(self):
        """
        Initialize the position to the starting platform.
        """
        # Define properties of player
        self.position = point(0, PLATFORM_HEIGHT)
        self.velocity = vector(0.0, 0.0)

        # To identify the start position, used for rendering
        self.start_position = self.position.copy()

        # Some constant for action
        self.hop_dy0 = 35.0
        self.leap_dy0 = 25.0

    def reset(self):
        """
        Reset the location of agent.
        """
        self.position = point(0, PLATFORM_HEIGHT)
        self.velocity = vector(0.0, 0.0)

    def update(self, dt):
        """
        Update the position and velocity of player.

        Parameters
        ----------
        dt (): time interval since last update
        """
        self.position += self.velocity * dt
        self.position[0] = bound(self.position[0], 0.0, MAX_WIDTH)
        self.velocity[0] *= self.velocity_decay

    def accelerate(self, accel, dt=DT):
        """
        Applies a power to the entity in direction theta.

        Parameters
        ----------
        accel (numpy.ndarray): A two elements numpy array,
        dt ():
        """
        accel = bound_vector(accel, MAX_DDX, MAX_DDY)
        # Update the velocity using acceleration and time
        self.velocity += accel * dt
        # Add noise to the velocity, same operation for agent.
        self.velocity[0] -= abs(np.random.normal(0.0, ENEMY_NOISE * dt))
        # Make sure the position is in valid range
        self.velocity = bound_vector(self.velocity, MAX_DX, MAX_DY)
        # Make sure the horizontal velocity is always positive, e.g.
        # Always move forward.
        self.velocity[0] = max(self.velocity[0], 0.0)

    def ground_bound(self):
        """
        Bound dx while on the ground.
        """
        self.velocity[0] = bound(self.velocity[0], 0.0, MAX_DX_ON_PLATFORM)

    def run(self, power, dt):
        """
        Run for a given power and time.
        """
        if dt > 0:
            self.accelerate(vector(power / dt, 0.0), dt)

    def jump(self, power):
        """
        Jump up for a single step.
        """
        self.accelerate(vector(0.0, power / DT))

    def jump_to(self, diffx, dy0, dev):
        """
        Jump to a specific position.
        """
        time = 2.0 * dy0 / GRAVITY + 1.0
        dx0 = diffx / time - self.velocity[0]
        dx0 = bound(dx0, -MAX_DDX, MAX_DY - dy0)
        if dev > 0:
            noise = -abs(np.random.normal(0.0, dev, 2))
        else:
            noise = np.zeros((2,))
        accel = vector(dx0, dy0) + noise
        self.accelerate(accel / DT)

    def hop_to(self, diffx):
        """
        Jump high to a position.
        """
        self.jump_to(diffx, self.hop_dy0, HOP_DEV)

    def leap_to(self, diffx):
        """
        Jump over a gap.
        """
        self.jump_to(diffx, self.leap_dy0, LEAP_DEV)

    def fall(self):
        """
        Apply gravity.
        """
        self.accelerate(vector(0.0, -GRAVITY))

    def decollide(self, other):
        """
        Shift overlapping entities apart.
        """
        precorner = other.position - self.size
        postcorner = other.position + other.size
        newx, newy = self.position[0], self.position[1]
        if self.position[0] < other.position[0]:
            newx = precorner[0]
        elif self.position[0] > postcorner[0] - self.size[0]:
            newx = postcorner[0]
        if self.position[1] < other.position[1]:
            newy = precorner[1]
        elif self.position[1] > postcorner[1] - self.size[1]:
            newy = postcorner[1]
        if newx == self.position[0]:
            self.velocity[1] = 0.0
            self.position[1] = newy
        elif newy == self.position[1]:
            self.velocity[0] = 0.0
            self.position[0] = newx
        elif abs(self.position[0] - newx) < abs(self.position[1] - newy):
            self.velocity[0] = 0.0
            self.position[0] = newx
        else:
            self.velocity[1] = 0.0
            self.position[1] = newy

    def above_platform(self, platform):
        """
        Checks the player is above the platform.
        """
        return -self.size[0] <= self.position[0] - platform.position[0] <= platform.size[0]

    def on_platform(self, platform):
        """
        Checks the player is standing on the platform.
        """
        on_y = self.position[1] - platform.position[1] == platform.size[1]
        return self.above_platform(platform) and on_y

    def colliding(self, other):
        """
        Check if two entities are overlapping.
        """
        precorner = other.position - self.size
        postcorner = other.position + other.size
        collide = (precorner < self.position).all()
        collide = collide and (self.position < postcorner).all()
        return collide


class PlatformWorld(object):
    """
    This class represent the environment.
    """

    def __init__(self):
        """
        The entities are setup and added to a space.
        """
        self.x_pos = 0.0
        self.player = Player()

        # Define platforms, we have three of them
        self.platforms = []
        for index in range(3):
            self.platforms.append(Platform(
                PLATFORMS_X[index], PLATFORMS_Y[index], PLATFORMS_WIDTH[index]))

        # Create enemies, we have two of them
        self.enemies = []
        for index in range(2):
            self.enemies.append(Agent(self.platforms[index]))

        # states
        self.states = []

    def reset(self):
        """
        Reset the position of enemies and player, start
        a new episodes.
        """
        for enemy in self.enemies:
            enemy.reset()
        self.player.reset()
        return self.get_state()

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
            pos = PLATFORMS_WIDTH[0] + GAP1
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
        return [wd1 / MAX_PLATWIDTH, wd2 / MAX_PLATWIDTH, gap / MAX_GAP, pos / MAX_WIDTH, diff / MAX_HEIGHT]


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
        end_episode = self.player.position[1] < self.lower_bound() + PLATFORM_HEIGHT
        right = self.player.position[0] >= self.right_bound()
        for entity in self.enemies:
            if self.player.colliding(entity):
                end_episode = True
        if right:
            reward = (self.right_bound() - self.x_pos) / self.right_bound()
            end_episode = True
        return reward, end_episode

    def update(self, action, dt=DT, interface=False):
        """
        Performs a single transition with the given action,
        then returns the new state and a reward.
        """
        if interface:
            self.xpos = self.player.position[0]
        self.states.append([self.player.position.copy(),
                            self.enemies[0].position.copy(),
                            self.enemies[1].position.copy()])
        self.perform_action(action, dt)
        if self.on_platforms():
            self.player.ground_bound()
        if self.player.position[0] > self.platforms[1].position[0]:
            enemy = self.enemies[1]
        else:
            enemy = self.enemies[0]
        for entity in [self.player, enemy]:
            entity.update(dt)
        for platform in self.platforms:
            if self.player.colliding(platform):
                self.player.decollide(platform)
                self.player.velocity[0] = 0.0
        reward = (self.player.position[0] - self.xpos) / self.right_bound()
        return self.terminal_check(reward)

    def take_action(self, action):
        ''' Take a full, stabilised update. '''
        end_episode = False
        run = True
        act, params = action
        self.xpos = self.player.position[0]
        step = 0
        difft = 1.0
        while run:
            if act == "run":
                reward, end_episode = self.update(('run', abs(params)), DT)
                difft -= DT
                run = difft > 0
            elif act in ['jump', 'hop', 'leap']:
                reward, end_episode = self.update(action)
                run = not self.on_platforms()
                action = None
            if end_episode:
                run = False
            step += 1
        state = self.get_state()
        return state, reward, end_episode, step
