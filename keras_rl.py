from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

import rl
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

import matplotlib.pyplot as plt
import cv2

mode = "train"

WINDOW_LENGTH = 1


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        return observation

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch.reshape((processed_batch.shape[0], 84, 84, 4))

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)



# Get the environment and extract the number of actions.
env = Atari("BreakoutNoFrameskip-v4", 10000, 110, 84, frame_skip=4)
np.random.seed(123)
env.env.seed(123)
nb_actions = env.action_space_size

# first layer takes in the 4 grayscale cropped image
input_lyr = keras.layers.Input((84, 84, 4), name="Input_last_4_frames")

# second layer convolves 16 8x8 then applies ReLU activation
x = keras.layers.Conv2D(16, (8,8), strides=4, name="Hidden_layer_1")(input_lyr)
x = keras.layers.Activation('relu')(x)

# third layer is the same but with 32 4x4 filters
x = keras.layers.Conv2D(32, (4,4), strides=2, name="Hidden_layer_2")(x)
x = keras.layers.Activation('relu')(x)

# James Note: Missing final dense hidden layer:
x = keras.layers.Flatten(name="Flatten")(x)
x = keras.layers.Dense(256, name="Hidden_layer_3")(x)
x = keras.layers.Activation('relu')(x)

# output layer is a fullyconnected linear layer
x = keras.layers.Dense(nb_actions, activation='linear')(x)

model = keras.Model(inputs=input_lyr, outputs=x, name="ATARI_DQN")


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=500_000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50_000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format("Breakout-v0")
    checkpoint_weights_filename = 'dqn_' + "Breakout-v0" + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format("Breakout-v0")
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=5)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format("Breakout-v0")
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)