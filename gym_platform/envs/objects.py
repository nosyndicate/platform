import numpy as np
from numpy.random import uniform

# Parameters for the platform
PLATFORMS_WIDTH = [250, 275, 50]
PLATFORM_HEIGHT = 40
PLATFORMS_X = [0.0, 475, 985]
PLATFORMS_Y = [0.0, 0.0, 0.0]

# Width of the gaps
GAPS = [225, 235]

# Max of something
MAX_WIDTH = sum(PLATFORMS_WIDTH + GAPS)
MAX_HEIGHT = max(1.0, max(PLATFORMS_Y))
MAX_PLATWIDTH = max(PLATFORMS_WIDTH)
MAX_GAP = max(GAPS)

# Duration of a time unit, some action may last several time units
DT = 0.05

# Max velocity on x-axis and y-axis
MAX_DX = 100.0
MAX_DY = 200.0
# Max x-axis velocity for player on the platform
MAX_DX_ON_PLATFORM = 70.0

# Max x-axis acceleration
MAX_DDX = (MAX_DX - MAX_DX_ON_PLATFORM) / DT
# Max y-axis acceleration
MAX_DDY = MAX_DY / DT

# Some other constants
ENEMY_SPEED = 30.0
LEAP_DEV = 1.0
HOP_DEV = 1.0
ENEMY_NOISE = 0.5
GRAVITY = 9.8


def point(x, y):
    """
    Return a numpy array represent a point in 2D surface.
    """
    return np.array([float(x), float(y)])


def vector(x, y):
    """
    Return a numpy array represent a vector in 2D space
    """
    return point(x, y)


def bound(value, lower_b, upper_b):
    """
    Clips off a value which exceeds the lower or upper bounds.

    Parameters
    ----------
    lower_b (float): lower bound of value
    upper_b (float): upper bound of value

    Returns
    -------
    value (float): clipped value in [lower_b, upper_b]
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
    Defines the agent (enemy).
    """
    size = vector(20.0, 30.0)

    def __init__(self, platform):
        """
        Initialize the agent on the platform.

        Parameters
        ----------
        platform (Platform): The platform this agent belong to.
        """
        self.platform = platform

        # Set the status of agent
        self.reset()

        # Set the leftmost and rightmost position the agent can be
        self.leftmost = self.platform.position[0]
        self.rightmost = self.platform.position[0] + self.platform.size[0] - self.size[0]

        # To identify the start position, used for rendering
        self.start_position = self.position.copy()

    def reset(self):
        """
        Reset the location of agent.
        """
        # dx is the velocity of enemy on x-axis
        self.dx = -ENEMY_SPEED
        # Set the the position of agent to the end of platform
        self.position = self.platform.size + self.platform.position
        # Due to size of agent, need to move it a little inside
        self.position[0] -= self.size[0]

    def update(self, dt):
        """
        Shift the agent along the platform.

        Parameters
        ----------
        dt (float): time interval of movement.
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
        # Set the status of player
        self.reset()

        # To identify the start position, used for rendering
        self.start_position = self.position.copy()

        # Some constants for different action
        self.hop_dy0 = 35.0
        self.leap_dy0 = 25.0

    def reset(self):
        """
        Reset the status of agent.
        """
        self.position = point(0, PLATFORM_HEIGHT)
        self.velocity = vector(0.0, 0.0)

    def update(self, dt):
        """
        Update the position and velocity of player.

        Parameters
        ----------
        dt (float): time interval since last update
        """
        # Update position according to velocity and time
        self.position += self.velocity * dt
        # Bound velocity on x-axis
        self.position[0] = bound(self.position[0], 0.0, MAX_WIDTH)
        # Velocity decay
        self.velocity[0] *= self.velocity_decay

    def accelerate(self, accel, dt=DT):
        """
        Applies a power to the entity in direction theta.

        Parameters
        ----------
        accel (numpy.ndarray): A two elements numpy array, specify the
            acceleration of x-axis and y-axis
        dt (float): time to perform acceleration
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

    def run(self, dx_change, dt):
        """
        Run for a given power and time.

        Parameters
        ----------
        dx_change (float): value add to velocity on x-axis, could be
            negative
        dt (float):
        """
        if dt > 0:
            self.accelerate(vector(dx_change / dt, 0.0), dt)

    def jump(self, power):
        """
        Jump up for a single step.
        """
        self.accelerate(vector(0.0, power / DT))

    def jump_to(self, diffx, dy0, dev):
        """
        Jump to a specific position.

        Parameters
        ----------
        diffx (float):
        dy0 (float): change of velocity along y-axis
        dev (float): if greater than 0, we add noise to
        """
        time = 2.0 * dy0 / GRAVITY + 1.0
        dx0 = diffx / time - self.velocity[0]
        dx0 = bound(dx0, -MAX_DDX, MAX_DY - dy0)
        if dev > 0:
            noise = -abs(np.random.normal(0.0, dev, 2))
        else:
            noise = np.zeros((2,))
        velocity_change = vector(dx0, dy0) + noise
        self.accelerate(velocity_change / DT)

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

        other (Agent):
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
        Checks the player is standing on (attached to) the platform.
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
