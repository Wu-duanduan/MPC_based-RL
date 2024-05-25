import gym
import random
import torch
import numpy as np
from ddpg_agent import Agent
import time


env = gym.make('Humanoid-v2')
# env = gym.make('RoboschoolHumanoid-v1')
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = Agent(state_size, action_size, random_seed=0)

agent.actor_local.load_state_dict(torch.load('5k_1k_input_noise_action.pth'))
agent.critic_local.load_state_dict(torch.load('5k_1k_input_noise_critic.pth'))

state = env.reset()
agent.reset()

while True:
    action = agent.act(state)
    env.render()
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        time.sleep(0.5)
        env.reset()

env.close()