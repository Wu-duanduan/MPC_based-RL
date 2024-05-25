# import roboschool
import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import rl_utils
from ddpg_agent import Agent
from save_model import Savemodel
from scipy.io import savemat

env = gym.make('Humanoid-v2')
# env = gym.make('RoboschoolHumanoid-v1')
num = 10
random.seed(num)
np.random.seed(num)
env.seed(num)
torch.manual_seed(num)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = Agent(state_size, action_size, random_seed=0)
savemodel = Savemodel(state_size, action_size)

def ddpg(n_episodes=2000, max_t=1000):
    scores_deque = deque(maxlen=20)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes + 1):
        # env.close()
        state = env.reset()
        # print(state)
        # exit()
        # agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            # state_all, action_all, reward_all, next_state_all = savemodel.save(state, action, reward, next_state)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            # env.render()

            if done:
                break

        if scores_deque:
            m = max(scores_deque)
            if (score > m):
                torch.save(agent.actor_local.state_dict(), '5k_1k_input_noise_action.pth')
                torch.save(agent.critic_local.state_dict(), '5k_1k_input_noise_critic.pth')

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score),
              end="")
        if i_episode % 20 == 0:
            # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            print(action)
    return scores


scores = ddpg()
# torch.save(state_all, 'state_all.pth')
# torch.save(action_all, 'action_all.pth')
# torch.save(reward_all, 'reward_all.pth')
# torch.save(next_state_all, 'next_state_all.pth')
# tensorboard logging

# writer = tf.summary.FileWriter('./graphs/5k_1k_input_noise')
# for k in range(len(scores)):
#     # step = [i+1 for i in range(len(scores))]
#
#     summary = tf.Summary(value=[tf.Summary.Value(simple_value=scores[k])])
#     writer.add_summary(summary, k + 1)

# tensorboard logging
episodes_list = list(range(len(scores)))
plt.plot(episodes_list, scores)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.show()

mv_return = rl_utils.moving_average(scores, 19)
# savemat('DDPG.mat', {'episodes_list': episodes_list, 'scores': scores, 'mv_return': mv_return})
np.save('episodes.npy', episodes_list)
np.save('mv_return04.npy', mv_return)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(1, len(scores) + 1), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()

# Testing Agent
# agent.actor_local.load_state_dict(torch.load('5k_1k_input_noise_action.pth'))
# agent.critic_local.load_state_dict(torch.load('5k_1k_input_noise_critic.pth'))
#
# state = env.reset()
# agent.reset()
# while True:
#     action = agent.act(state)
#     env.render()
#     next_state, reward, done, _ = env.step(action)
#     state = next_state
#     if done:
#         env.reset()
#
# env.close()
