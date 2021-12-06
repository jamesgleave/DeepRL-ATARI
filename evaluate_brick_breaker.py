from src import atari
from src import deep_q_agent
from src import deep_q_network, playground
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ours", "transfer"])
    parser.add_argument("--games", required=False, default=1, type=int)
    parser.add_argument("--render", required=False, default=False, action="store_true")

    args = parser.parse_args()

    # Create the game
    game = atari.Atari("BreakoutNoFrameskip-v4", 84, 84, frame_skip=4, clip_reward=False)

    if args.model == "transfer":
        weight_path = "src/extras/logs/dqn_model_og.h5"
        model_config = "2015"
        eps = 0.01

    elif args.model == "ours":
        weight_path = "src/extras/logs/main_model_weights.h5f"
        model_config = "2013"

        # This flips the order of the frames in the 84x84x4 image
        game.temporal_flip = True

        # Seed for reproduceability
        np.random.seed(42)

        # Set the epsilon like the paper
        eps = 0.05

    # Setup the default values that we use
    BATCH_SIZE = 32
    MIN_EPSILON = 0.1
    EPS_DECAY = 0.9/500_000 # 0.9/1_000_000 (Half of the eps)

    network = deep_q_network.DeepQNetwork(game.action_space_size, 0.00085, BATCH_SIZE, model_config=model_config)
    network.Model.summary()
    agent = deep_q_agent.DeepQAgent(game=game,
                                    model=network,
                                    gamma=0.99,
                                    epsilon=1,
                                    min_epsilon=MIN_EPSILON,
                                    epsilon_decay=EPS_DECAY,
                                    replay_memory_size=500_000,
                                    exploration_steps=100_000,
                                    target_update_horizon=10_000,
                                    main_model_train_horizon=4,
                                    min_replay_memory_size=BATCH_SIZE,
                                    save_frequency=250)

    network.Model.load_weights(weight_path)
    mean_rewards = agent.evaluate(epsilon=eps, n_games=args.games, render=args.render)
    print(f"Mean Rewards: {mean_rewards}")

