import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 1. Environment & Parameters
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0] 
act_dim = env.action_space.n # discrete, 0: left, 1:right

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
lr = 1e-3
batch_size = 32

# 2. Q Network
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.fc(x)
# state = [s1, s2, s3, s4]
# Q(state) = [Q(state, action=0)]


q_net = QNet()
target_net = QNet()
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# 3. Replay buffer
memory = deque(maxlen=5000)

# 4. Training
state, _ = env.reset()
for step in range(5000):
    # epsilon greedy choose actions
    if random.random()<epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            q = q_net(torch.tensor(state, dtype = torch.float32))
            action = torch.argmax(q).item()
        
    next_state, reward, done, truncated, _ = env.step(action)
    # Replay Buffer
    memory.append((state, action, reward, next_state, done or truncated))
    state = next_state

    if done or truncated:
        state, _ = env.reset()

    # Learning
    if len(memory)>=batch_size:
        batch = random.sample(memory, batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(s2, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)
    
        # Q(s, a)
        q_sa = q_net(s).gather(1, a)

        # target = r+gamma*max Q_target(s')
        with torch.no_grad():
            max_q_next = target_net(s2).max(1, keepdim=True)[0]
            # target = r+gamma*max Q_target(s') s' is the next statement
            target = r+gamma*(1-d)*max_q_next
        loss = loss_fn(q_sa, target) # let Q network satisfied Bellman equations

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # update target network
    if step%1000==0:
        target_net.load_state_dict(q_net.state_dict())

    # episilon decay
    epsilon = max(epsilon_min, epsilon*epsilon_decay)

    if step%10==0:
        print(f"step {step}, epsilon = {epsilon:.3f}")
env.close()