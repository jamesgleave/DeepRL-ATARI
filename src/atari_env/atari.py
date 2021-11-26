import numpy as np
import gym
import cv2 
import matplotlib.pyplot as plt


class Atari(object):
    def __init__(self, game: str, max_steps: int, resized_width: int, resized_height: int):
        self.env = gym.make(game)
        self.env.reset()
        self.max_steps = max_steps # Max steps per episode (int)
        self.action_space_size = self.env.action_space # The number of actions an agent can perform (int)
        self.resized_width = resized_width
        self.resized_height = resized_height
        
    def reset(self, **kwargs):
        # Should reset the environment to the begining
        # Returns initial state
        self.env.reset()

    def num_actions_available(self):
        # Return total number of actions
        return self.env.action_space.n

    def get_preprocessed_frame(self, observation):
        # Convert image to grayscale
        # Rescale image
        image = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.resized_height, self.resized_width))
        return image[26:, :]

    def print_action_meanings(self):
        # Prints meaings of all possible actions
        print(self.env.get_action_meanings())


    def step(self, action: int):
        # take a step in the env and return the next state, reward,
        # and if the game is done as a tuple
        observation, reward, done, lives = self.env.step(action)
        
        # Step function to set rewards to -1, +1 or 0
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        else:
            reward = 0 
            
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
print(game.step(0))