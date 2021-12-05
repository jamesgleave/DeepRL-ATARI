
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

#Evaluate agents performance, in a number of games
def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    s = env.reset()
    for _ in range(n_games):
        reward = 0
        for t in range(t_max):
            # if t == 100:
            #     plt.subplot(2,2,1)
            #     plt.imshow(s[:,:,0])
            #     plt.subplot(2,2,2)
            #     plt.imshow(s[:,:,1])
            #     plt.subplot(2,2,3)
            #     plt.imshow(s[:,:,2])
            #     plt.subplot(2,2,4)
            #     plt.imshow(s[:,:,3])
            #     plt.show()
            #     raise Exception
            # env.render()
            # time.sleep(0.01)
            qvalues = agent.main_model.predict(np.asarray([s]))
            action = qvalues.argmax(axis=-1)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done: 
              s = env.reset()
              break
        rewards.append(reward)
    return np.mean(rewards)

from src.deep_q_network.deep_q_network import DeepQNetwork
from src.deep_q_agent.deep_q_agent import DeepQAgent

if __name__ == '__main__':
    env = gym.make("BreakoutDeterministic-v4")
    env = FrameBuffer(env, n_frames=4, img_size=(84,84))
    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    model = DeepQNetwork(4, weights='dqn_model_og.h5')
    agent = DeepQAgent(game=env,
                        model=model,
                        gamma=0.99,
                        alpha=1,
                        epsilon=0.001,
                        min_epsilon=0.001,
                        replay_memory_size=100_000,
                        exploration_steps=100,
                        target_update_horizon=10_000,
                        main_model_train_horizon=4,
                        min_replay_memory_size=32
    )

    print(evaluate(env, agent, n_games=1))