import numpy as np
import gym
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import io 


class Atari(object):
    def __init__(self, game: str, max_steps: int, resized_width: int, resized_height: int):
        self.env = gym.make(game)
        self.env.reset()
        self.max_steps = max_steps # Max steps per episode (int)
        self.action_space_size = self.env.action_space # The number of actions an agent can perform (int)
        self.resized_width = resized_width
        self.resized_height = resized_height
        #self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        
    def reset(self, **kwargs):
        # Should reset the environment to the begining
        # Returns initial state
        self.env.reset()

    def num_actions_available(self):
        # Return total number of actions
        return self.env.action_space.n

    def get_preprocessed_frame(self, observation):
        #1) Get image grayscale
        #2) Rescale image
        image = resize(rgb2gray(observation), (self.resized_width, self.resized_height))
        return image[26:, :]

    def print_action_meanings(self):
        print(self.env.get_action_meanings())


        # -> tuple
    def step(self, action: int):
        # take a step in the env and return the next state, reward,
        # and if the game is done as a tuple
        observation, reward, done, lives = self.env.step(action)
        image = self.get_preprocessed_frame(observation)
        print(image.shape)
        plt.imshow(image, cmap="gray")
        plt.show()
        return image, reward, done, lives
    
    def render(self):
        # Render the game state
        for _ in range(self.max_steps):
            self.env.render()
            self.step(2)

game = Atari("Breakout-v0", 10000, 110, 84)

print(game.reset())