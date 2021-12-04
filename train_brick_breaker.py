from src import atari
from src import deep_q_agent
from src import deep_q_network
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 32
MIN_EPSILON = 0.1
EPS_DECAY = 0.9/250_000 # 0.9/1_000_000

game = atari.Atari("BreakoutNoFrameskip-v4", 10000, 110, 84, frame_skip=4)
network = deep_q_network.DeepQNetwork(game.action_space_size, 0.00085, BATCH_SIZE)
agent = deep_q_agent.DeepQAgent(game=game,
                                model=network,
                                gamma=0.99,
                                alpha=1,
                                epsilon=1,
                                min_epsilon=MIN_EPSILON,
                                epsilon_decay=EPS_DECAY,
                                replay_memory_size=100_000,
                                exploration_steps=5,
                                target_update_horizon=10_000,
                                main_model_train_horizon=4,
                                min_replay_memory_size=BATCH_SIZE,)


agent.train(max_episodes=10_000, max_frames=1_000_000,show_game=True)
