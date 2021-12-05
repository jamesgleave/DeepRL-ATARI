
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

# Gym imports
from gym.core import ObservationWrapper
from gym.core import Wrapper
from gym.spaces.box import Box
import time
import gym

from src.deep_q_network.deep_q_network import DeepQNetwork
from src.deep_q_agent.deep_q_agent import DeepQAgent
from src.atari_env.atari import Atari

class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4, img_size=(84, 84)):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)
        self.img_size = img_size

        self.obs_shape = [img_size[0], img_size[1], n_frames] # 1 channel for each fram (grayscaled)
        self.framebuffer = np.zeros(self.obs_shape, 'float32')
        self.action_space_size = self.env.action_space.n

    def process_obs(self, observation):
        """what happens to each observation"""
        
        # crop image (top and bottom, top from 34, bottom remove last 16)
        img = observation[34:-16, :, :]
        
        # resize image
        img = cv2.resize(img, self.img_size)
        
        img = img.mean(-1, keepdims=True)
        
        img = img.astype('float32') / 255.
        return img

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.process_obs(self.env.reset()))
        return self.framebuffer
    
    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(self.process_obs(new_img))
        return self.framebuffer, reward, done, info
    
    def update_buffer(self, img):
        offset = 1 # 1 channel per pixel
        axis = -1
        cropped_framebuffer = self.framebuffer[:,:,:-offset]
        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis=axis)


if __name__ == '__main__':
    ours = True
    if ours:
        env =Atari("BreakoutNoFrameskip-v4", 84, 84, frame_skip=4)
    else:
        env = gym.make("BreakoutDeterministic-v4")
        env = FrameBuffer(env, n_frames=4, img_size=(84,84))
    env.reset()

    model = DeepQNetwork(4, weights='dqn_model_og.h5')
    agent = DeepQAgent(game=env,
                        model=model,
                        gamma=0.99,
                        epsilon=0.001,
                        min_epsilon=0.001,
                        replay_memory_size=100_000,
                        exploration_steps=100,
                        target_update_horizon=10_000,
                        main_model_train_horizon=4,
                        min_replay_memory_size=32
    )

    print(agent.evaluate(epsilon=0.001, n_games=1))