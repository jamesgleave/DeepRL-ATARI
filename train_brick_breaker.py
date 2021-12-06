from src import atari
from src import deep_q_agent
from src import deep_q_network
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 32
MIN_EPSILON = 0.1
EPS_DECAY = 0.9/500_000 # 0.9/1_000_000 (Half of the eps)

game = atari.Atari("BreakoutNoFrameskip-v4", 84, 84, frame_skip=4)
network = deep_q_network.DeepQNetwork(game.action_space_size, 0.00025, BATCH_SIZE)
network.compile()
agent = deep_q_agent.DeepQAgent(game=game,
                                model=network,
                                gamma=0.99,
                                epsilon=1,
                                min_epsilon=MIN_EPSILON,
                                epsilon_decay=EPS_DECAY,
                                replay_memory_size=500_000,
                                exploration_steps=50_000,
                                target_update_horizon=10_000,
                                main_model_train_horizon=4,
                                min_replay_memory_size=BATCH_SIZE,
                                save_frequency=250)


agent.train(max_episodes=100_000, max_frames=2_500_000, show_game=False)