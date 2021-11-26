
# 1. It renders instances for 500 timesteps, performing random actions.
import gym
env = gym.make('Breakout-v0')
env.reset()
for _ in range(500):
    env.render()
    env.step(env.action_space.sample())
# 2. To check all env available, uninstalled ones are also shown.

from gym import envs
print(envs.registry.all())