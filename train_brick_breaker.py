from src import atari
from src import deep_q_agent
from src import deep_q_network, playground
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 32
MIN_EPSILON = 0.1
EPS_DECAY = 0.9/250_000 # 0.9/1_000_000

game = atari.Atari("BreakoutNoFrameskip-v4", 50, 84, 84, frame_skip=4)
network = deep_q_network.DeepQNetwork(game.action_space_size, 0.00085, BATCH_SIZE)
# agent = deep_q_agent.DeepQAgent(game=game,
#                                 model=network,
#                                 gamma=0.99,
#                                 alpha=1,
#                                 epsilon=1,
#                                 min_epsilon=MIN_EPSILON,
#                                 epsilon_decay=EPS_DECAY,
#                                 replay_memory_size=100_000,
#                                 exploration_steps=100,
#                                 target_update_horizon=10_000,
#                                 main_model_train_horizon=4,
#                                 min_replay_memory_size=BATCH_SIZE,)


# agent.train(max_episodes=10_000, max_frames=1_000_000,show_game=True)

#Evaluate agents performance, in a number of games
def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    s = env.reset()
    for _ in range(n_games):
        reward = 0
        for t in range(t_max):
            if t == 100:
                plt.subplot(2,2,1)
                plt.imshow(s[:,:,0])
                plt.subplot(2,2,2)
                plt.imshow(s[:,:,1])
                plt.subplot(2,2,3)
                plt.imshow(s[:,:,2])
                plt.subplot(2,2,4)
                plt.imshow(s[:,:,3])
                plt.show()
                raise Exception
            qvalues = agent.get_qvalues([s])
            action = agent.sample_actions(qvalues)[0] # epsilon greedy
            s, r, done, _ = env.step(action)
            reward += r
            env.env.render()
            if done:
              s = env.reset()
              break
        rewards.append(reward)
    return np.mean(rewards)
print(evaluate(game, network))