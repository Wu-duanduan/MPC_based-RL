import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from scipy.io import savemat

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class S_net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(S_net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, state_dim)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class Rnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Rnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.s_net = S_net(state_dim, hidden_dim, self.action_dim).to(device)  # S+1网络
        self.r_net = Rnet(state_dim, hidden_dim, self.action_dim).to(device)  # R网络
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.optimizer2 = torch.optim.Adam(self.s_net.parameters(), lr=critic_lr)
        self.optimizer3 = torch.optim.Adam(self.r_net.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q0_values = self.critic(states, actions)  # Q值
        s1_1_values = torch.reshape(self.s_net(states, actions)[:, 0], [batch_size, 1])  # S+1值
        s1_2_values = torch.reshape(self.s_net(states, actions)[:, 1], [batch_size, 1])  # S+1值
        s1_3_values = torch.reshape(self.s_net(states, actions)[:, 2], [batch_size, 1])  # S+1值
        s1_values = torch.cat((s1_1_values, s1_2_values, s1_3_values), 1)
        r0_values = self.r_net(states, actions)  # R值

        a1_values = self.actor(s1_values)
        a1_values = torch.reshape(a1_values, [batch_size, 1])

        q1_values = self.critic(s1_values, a1_values)  # Q+1值
        s2_1_values = torch.reshape(self.s_net(s1_values, a1_values)[:, 0], [batch_size, 1])  # S+2值
        s2_2_values = torch.reshape(self.s_net(s1_values, a1_values)[:, 1], [batch_size, 1])  # S+2值
        s2_3_values = torch.reshape(self.s_net(s1_values, a1_values)[:, 2], [batch_size, 1])  # S+2值
        s2_values = torch.cat((s2_1_values, s2_2_values, s2_3_values), 1)
        r1_values = self.r_net(s1_values, a1_values)  # R+1值

        a2_values = self.actor(s2_values)
        a2_values = torch.reshape(a2_values, [batch_size, 1])

        q2_values = self.critic(s2_values, a2_values)  # Q+1值
        s3_1_values = torch.reshape(self.s_net(s2_values, a2_values)[:, 0], [batch_size, 1])   # S+2值
        s3_2_values = torch.reshape(self.s_net(s2_values, a2_values)[:, 1], [batch_size, 1])  # S+2值
        s3_3_values = torch.reshape(self.s_net(s2_values, a2_values)[:, 2], [batch_size, 1])   # S+2值
        s3_values = torch.cat((s3_1_values, s3_2_values, s3_3_values), 1)
        r2_values = self.r_net(s2_values, a2_values)  # R+1值

        next_q1_values = self.target_critic(next_states, self.target_actor(next_states))
        q0_targets = rewards + self.gamma * next_q1_values * (1 - dones)
        next_q2_values = self.target_critic(s2_values, self.target_actor(s2_values))
        q1_targets = r1_values + self.gamma * next_q2_values * (1 - dones)
        next_q3_values = self.target_critic(s3_values, self.target_actor(s3_values))
        q2_targets = r2_values + self.gamma * next_q3_values * (1 - dones)

        critic_loss = torch.mean(F.mse_loss(q0_values, q0_targets) + F.mse_loss(q1_values, q1_targets)
                                 + F.mse_loss(q2_values, q2_targets))  # 均方误差损失函数
        dqn2_loss = torch.mean(F.mse_loss(s1_values, next_states))
        dqn3_loss = torch.mean(F.mse_loss(r0_values, rewards))

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        #
        self.optimizer2.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn2_loss.backward()  # 反向传播更新参数
        self.optimizer2.step()
        #
        self.optimizer3.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn3_loss.backward()  # 反向传播更新参数
        self.optimizer3.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络
        return dqn2_loss.detach().numpy(), dqn3_loss.detach().numpy()
actor_lr = 3e-4
critic_lr = 3e-3
num_episodes = 200
hidden_dim = 64
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 5000
minimal_size = 1000
batch_size = 64
sigma = 0.01  # 高斯噪声标准差
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'Pendulum-v0'
env = gym.make(env_name)
seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
env.seed(seed_num)
torch.manual_seed(seed_num)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
s_a_dim = action_dim * state_dim
action_bound = env.action_space.high[0]  # 动作最大值
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

return_list, s_loss_list, r_loss_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
# savemat('DDPG_MPC3.mat', {'episodes_list': episodes_list, 'return_list': return_list, 'mv_return': mv_return, 's_loss': s_loss_list, 'r_loss': r_loss_list})
np.save('mv_return71.npy', mv_return)
np.save('s_loss31.npy', s_loss_list)
np.save('r_loss31.npy', r_loss_list)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()
