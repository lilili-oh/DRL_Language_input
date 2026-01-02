import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden=128, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim=4, lr=5e-4, gamma=0.99, device=None, distance_coef=0.0):
        """DQN agent.

        distance_coef: float
            coefficient for including negative next-step distance-to-goal in TD target.
            If 0.0 (default), distance is not used. Positive values encourage moves
            that reduce distance to goal.
        """
        self.device = device or (torch.device('mps') if torch.cuda.is_available() else torch.device('cpu'))
        self.action_dim = action_dim
        self.gamma = gamma
        self.distance_coef = float(distance_coef)
        self.q_net = QNetwork(state_dim, n_actions=action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions=action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(20000)
        self.update_count = 0

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(s)
            return int(q.argmax().item())

    def push_transition(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def update(self, batch_size=64, tau=0.005):
        if len(self.replay) < batch_size:
            return 0.0
        trans = self.replay.sample(batch_size)
        state = torch.FloatTensor(np.array(trans.state)).to(self.device)
        action = torch.LongTensor(np.array(trans.action)).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(np.array(trans.reward)).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(trans.next_state)).to(self.device)
        done = torch.FloatTensor(np.array(trans.done).astype(np.float32)).unsqueeze(1).to(self.device)

        # predicted Q for taken actions
        q_values = self.q_net(state).gather(1, action)

        # Double DQN target: select action with online network, evaluate with target network
        # 在 update 前，假设 next_state 和 state 是 torch tensors
        # 计算 Phi(s) = -alpha * dist(s,goal)
        alpha = self.distance_coef  # reuse parameter, but small (~0.1)
        agent_pos = state[:, 0:2]
        goal_pos  = state[:, 2:4]
        dist_s = torch.norm(goal_pos - agent_pos, dim=1, keepdim=True)

        agent_pos_n = next_state[:, 0:2]
        goal_pos_n  = next_state[:, 2:4]
        dist_s_n = torch.norm(goal_pos_n - agent_pos_n, dim=1, keepdim=True)

        phi_s  = - alpha * dist_s
        phi_s_n = - alpha * dist_s_n

        # shaped reward
        shaped_reward = reward + (self.gamma * phi_s_n - phi_s)

        # next_q computed by Double DQN as before
        with torch.no_grad():
            next_actions = self.q_net(next_state).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_state).gather(1, next_actions)

        target = shaped_reward + (1.0 - done) * self.gamma * next_q
        loss = nn.SmoothL1Loss()(q_values, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.update_count += 1
        # soft update
        if tau < 1.0:
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        else:
            # hard update every 100 updates
            if self.update_count % 100 == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
