
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

from src import atari
from src import deep_q_agent
from src import deep_q_network, playground
import numpy as np
import matplotlib.pyplot as plt


class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self,env)

        self.img_size = (84, 84)
        self.observation_space = Box(0.0, 1.0, (self.img_size[0], self.img_size[1], 1))

    def observation(self, img):
        """what happens to each observation"""

        # crop image (top and bottom, top from 34, bottom remove last 16)
        img = img[34:-16, :, :]

        # resize image
        img = cv2.resize(img, self.img_size)

        img = img.mean(-1,keepdims=True)

        img = img.astype('float32') / 255.
        return img

class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4, dim_order='tensorflow'):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)
        self.dim_order = dim_order

        height, width, n_channels = env.observation_space.shape
        obs_shape = [height, width, n_channels * n_frames]

        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info

    def update_buffer(self, img):
        offset = self.env.observation_space.shape[-1]
        axis = -1
        cropped_framebuffer = self.framebuffer[:,:,:-offset]
        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis=axis)

class DeepQNetwork(object):
    def __init__(self, n_actions, epsilon=0):
        """
        An implementation of the exact network used in the Atari paper. So not many arguments needed.

        Args:
            num_actions (int): The number of possible actions. This will define the output shape of the model.
            learning_rate (float, optional): Defaults to 0.1.
            batch_size (int, optional): Defaults to the size that is used in the paper.
        """
        self.network = self.__build_model(n_actions)
        self.epsilon = epsilon

    @staticmethod
    def __build_model(num_actions:int) -> tf.keras.Model:
        """
        This function builds the exact DQN model from the Atari paper.
            Input shape is (84, 84, 4).
                - This is the preprocessed images of the last 4 frames in the history

            1st Hidden layer convolves 16 8x8 filters with stride 4.
                - followed by rectifier nonlinear

            2nd hidden layer convolves 32 4x4 filters with stride 2.
                - again followed by rectifier nonlinearity

            Output layer is a fully connected linear layer.
                - shape -> (a, ) where a is the number of actions
                - the ouput corresponds to the predicted Q-values

        Args:
            num_actions (int): this determines the output shape

        Returns:
            tf.keras.model: The DQN model from the paper
        """
        # first layer takes in the 4 grayscale cropped image
        input_lyr = tf.keras.layers.Input((84,84,4), name="Input_last_4_frames")

        # convolutional layers
        x = tf.keras.layers.Conv2D(32, (8,8), activation='relu', strides=4, use_bias=False, input_shape=(84,84,4), name="Hidden_layer_1")(input_lyr)
        x = tf.keras.layers.Conv2D(64, (4,4), activation='relu', strides=2, use_bias=False, name="Hidden_layer_2")(x)
        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', strides=1, use_bias=False, name="Hidden_layer_3")(x)
        x = tf.keras.layers.Conv2D(1024, (7,7), activation='relu', strides=1, use_bias=False, name="Hidden_layer_4")(x)

        # flattening for dense output
        x = tf.keras.layers.Flatten(name="Final_flatten")(x)
        x = tf.keras.layers.Dense(num_actions, activation='linear')(x)

        return tf.keras.Model(inputs=input_lyr, outputs=x, name="ATARI_DQN")

    def get_qvalues(self, state_t):
        return self.network.predict(np.asarray(state_t))

    def sample_actions(self, qvalues):
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p = [1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

#Evaluate agents performance, in a number of games
def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    env2 = atari.Atari("BreakoutNoFrameskip-v4", 84, 84, frame_skip=4)
    s = env2.reset()
    for _ in range(n_games):
        reward = 0
        for t in range(t_max):
            # Reverse that shit boi
            s = s[:, :, ::-1]

            # f, axarr = plt.subplots(4,2)
            # axarr[0,0].imshow(s[:, :, 0])
            # axarr[0,1].imshow(s[:, :, 1])
            # axarr[1,0].imshow(s[:, :, 2])
            # axarr[1,1].imshow(s[:, :, 3])



            # axarr[2,0].imshow((s2[:, :, 0] / 255.0).astype(np.float32))
            # axarr[2,1].imshow((s2[:, :, 1] / 255.0).astype(np.float32))
            # axarr[3,0].imshow((s2[:, :, 2] / 255.0).astype(np.float32))
            # axarr[3,1].imshow((s2[:, :, 3] / 255.0).astype(np.float32))
            # plt.show()
            # env2.render()
            # time.sleep(0.01)



            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env2.step(action)
            reward += r
            if done:
              s = env2.reset()
              break
        rewards.append(reward)
    return np.mean(rewards)


if __name__ == '__main__':
    #Instatntiate gym Atari-Breakout environment
    env = gym.make("BreakoutDeterministic-v4")
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4, dim_order='tensorflow')

    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    agent = DeepQNetwork(n_actions, epsilon=0.5)
    agent.network.load_weights('dqn_model_og.h5')
    agent.epsilon = 0.001
    print(evaluate(env, agent, n_games=5))


