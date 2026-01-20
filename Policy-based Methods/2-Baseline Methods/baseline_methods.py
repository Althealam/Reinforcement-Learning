# Background: DQN is not available for the scenario of continuous action space
# so the core of policy-based is learning the probability of action a in statement s pi_theta(a|s)

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 1. Policy Network
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 2. Initialization
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

policy = PolicyNet(obs_dim, act_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
gamma = 0.99

# 3. Training (REINFORCE)
for episode in range(1000):
    state, _ = env.reset()
    log_probs = []
    rewards = []

    # get the samples for a whole episode
    while True:
        state_t = torch.tensor(state, dtype = torch.float32)

        logits = policy(state_t)

        dist = Categorical(logits = logits)

        action = dist.sample()

        log_probs.append(dist.log_prob(action))

        next_state, reward, done, truncated, _ = env.step(action.item())

        rewards.append(reward)

        state = next_state

        if done or truncated:
            break
    
    # get the return G_t
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r+gamma*G
        returns.insert(0, G)

    returns = torch.tensor(returns)


    ### !update for baseline methods ###
    baseline = returns.mean() # baseline b
    advantages = returns - baseline # A_t = G_t-b
    ########

    # Update using advantage function
    loss = 0
    for log_prob, adv in zip(log_probs, advantages):
        loss += -log_prob*adv # use the advantage
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode%50==0:
        print(f"Episide {episode}, total reward={sum(rewards)}")

env.close()