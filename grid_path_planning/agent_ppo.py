import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

Transition = namedtuple('Transition', ('state','action','reward','next_state','done','logp'))

# ========== 网络 ==========
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden=128, n_actions=4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden), 
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
)


    def forward(self, x):
        f = self.feature(x)
        return self.policy(f), self.value(f)

    def act(self, state, device):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits, value = self.forward(s)
        probs = torch.softmax(logits, dim=1)
        dist  = torch.distributions.Categorical(probs)
        a = dist.sample()
        return a.item(), dist.log_prob(a), value.item()

# ========== PPO Agent ==========
class PPOAgent:
    def __init__(self, state_dim, action_dim=4, lr=3e-4, gamma=0.99, eps_clip=0.4, device=None):
        self.device = device or (torch.device('mps') if torch.cuda.is_available() else torch.device('cpu'))
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.net = ActorCritic(state_dim, n_actions=action_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, state):
        a, logp, v = self.net.act(state, self.device)
        return a, logp.detach(), v

    def compute_returns(self, rewards, dones, last_value):
        R = last_value
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.FloatTensor(returns)

    def update(self, trajectory, K_epochs=4):
        states  = torch.FloatTensor(np.array([t.state for t in trajectory])).to(self.device)
        actions = torch.LongTensor([t.action for t in trajectory]).to(self.device)
        rewards = [t.reward for t in trajectory]
        dones   = [t.done for t in trajectory]
        old_logp= torch.stack([t.logp for t in trajectory]).to(self.device)

        # 计算 returns & advantage
        last_state = trajectory[-1].next_state
        with torch.no_grad():
            _, last_v = self.net(states[-1].unsqueeze(0))
            last_v = last_v.item()

        returns = self.compute_returns(rewards, dones, last_v).to(self.device)
        adv = returns - self.net(states)[1].squeeze(1).detach()

        # PPO 训练循环
        for _ in range(K_epochs):
            logits, values = self.net(states)
            values = values.squeeze(1)
            dist = torch.distributions.Categorical(torch.softmax(logits, dim=1))
            logp = dist.log_prob(actions)

            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = nn.SmoothL1Loss()(values, returns)
            loss = actor_loss + 0.5 * critic_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

