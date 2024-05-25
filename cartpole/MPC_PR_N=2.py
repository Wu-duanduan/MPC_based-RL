import random
import gym
import numpy as np
import collections

import self as self
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from scipy.io import savemat

class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class SR_net(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(SR_net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + 1, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, state_dim + 1)
        # self.fc3 = torch.nn.Linear(hidden_dim, state_dim + 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))  # 隐藏层使用ReLU激活函数
        # x = F.relu(self.fc2(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

class DQN:
    ''' DQN算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 模型网络
        self.sr_net = SR_net(state_dim, hidden_dim,
                           self.action_dim).to(device)  # S+1网络
        # 使用Adam优化器
        self.optimizer1 = torch.optim.Adam(self.q_net.parameters(),
                                           lr=learning_rate)
        self.optimizer2 = torch.optim.Adam(self.sr_net.parameters(),
                                           lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q0_values = self.q_net(states).gather(1, actions)  # Q值
        s1_1_values = torch.reshape(self.sr_net(states, actions)[:, 0], [batch_size, 1])  # S+1值
        s1_2_values = torch.reshape(self.sr_net(states, actions)[:, 1], [batch_size, 1])  # S+1值
        s1_3_values = torch.reshape(self.sr_net(states, actions)[:, 2], [batch_size, 1])  # S+1值
        s1_4_values = torch.reshape(self.sr_net(states, actions)[:, 3], [batch_size, 1])  # S+1值
        s1_values = torch.cat((s1_1_values, s1_2_values, s1_3_values, s1_4_values), 1)
        r0_values = torch.reshape(self.sr_net(states, actions)[:, 4], [batch_size, 1])   # R值

        a1_values = self.q_net(s1_values).argmax(dim=1)
        a1_values = torch.reshape(a1_values, [batch_size, 1])

        q1_values = self.q_net(s1_values).gather(1, a1_values)  # Q+1值
        s2_1_values = torch.reshape(self.sr_net(s1_values, a1_values)[:, 0], [batch_size, 1])  # S+2值
        s2_2_values = torch.reshape(self.sr_net(s1_values, a1_values)[:, 1], [batch_size, 1])  # S+2值
        s2_3_values = torch.reshape(self.sr_net(s1_values, a1_values)[:, 2], [batch_size, 1])  # S+2值
        s2_4_values = torch.reshape(self.sr_net(s1_values, a1_values)[:, 3], [batch_size, 1])  # S+2值
        s2_values = torch.cat((s2_1_values, s2_2_values, s2_3_values, s2_4_values), 1)
        r1_values = torch.reshape(self.sr_net(s1_values, a1_values)[:, 4], [batch_size, 1])  # R+1值
        # 下个状态的最大Q值
        max_next_q1_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q0_targets = rewards + self.gamma * max_next_q1_values * (1 - dones
                                                                  )  # TD误差目标
        max_next_q2_values = self.target_q_net(s2_values).max(1)[0].view(
            -1, 1)
        q1_targets = r1_values + self.gamma * max_next_q2_values * (1 - dones
                                                                    )  # TD误差目标
        dqn1_loss = torch.mean(F.mse_loss(q0_values, q0_targets) + F.mse_loss(q1_values, q1_targets))  # 均方误差损失函数
        dqn2_loss = torch.mean(F.mse_loss(s1_values, next_states) + F.mse_loss(r0_values, rewards))
        #
        self.optimizer1.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn1_loss.backward(retain_graph=True)  # 反向传播更新参数
        self.optimizer1.step()
        #
        self.optimizer2.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn2_loss.backward()  # 反向传播更新参数
        self.optimizer2.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1
        return dqn2_loss.detach().numpy()


lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 5000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
env.seed(seed_num)
torch.manual_seed(seed_num)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

return_list = []
sr_loss_list = []
for i in range(20):
    with tqdm(total=int(num_episodes / 20), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 20)):
            episode_return = 0
            sr_loss = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    dqn2_loss = agent.update(transition_dict)
                    sr_loss += dqn2_loss
            return_list.append(episode_return)
            sr_loss_list.append(sr_loss)
            if (i_episode + 1) % 5 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 20 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
# savemat('DQN_MPC2.mat', {'episodes_list': episodes_list, 'return_list': return_list, 'mv_return': mv_return, 'sr_loss': sr_loss_list})
# np.save('episodes.npy', episodes_list)
# np.save('mv_return22.npy', mv_return)
# np.save('sr_loss22.npy', sr_loss_list)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()