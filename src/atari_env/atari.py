import numpy as np


class Atari(object):
    def __init__(self, **kwargs):

        # What agent needs
        self.max_steps = None # Max steps per episode (int)
        self.action_space_size = None # The number of actions an agent can perform (int)

    def reset(self, **kwargs) -> np.array:
        # Should reset the environment to the begining
        # Returns initial state
        raise NotImplementedError

    def step(self, action: int) -> tuple:
        # take a step in the env and return the next state, reward,
        # and if the game is done as a tuple
        raise NotImplementedError

    def render(self):
        # Render the game state
        raise NotImplementedError

    def get_image(self):
        # return an image of the game state
        raise NotImplementedError
