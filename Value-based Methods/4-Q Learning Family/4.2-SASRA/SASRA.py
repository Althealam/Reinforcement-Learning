import gymnasium as gym
import numpy as np
from collections import defaultdict

def sarsa(env, num_episodes = 5000, alpha = 0.1, gamma = 0.9, epsilon = 0.1):
    """
    SASRA algorithm (on-policy TD control)
    Args:
        env: Gymnasium environment
        num_episodes: number of training episode
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate for epislon-greedy policy
    Returns:
        Q: learned action-value function
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()

        # epsilon greedy action selection for current state
        if np.random.rand()<epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Choose next action based on current policy (on-policy)
            if not done:
                if np.random.rand()<epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = np.argmax(Q[next_state])
            else:
                next_action = None
            

            # TD target and update
            td_target = reward+(gamma*Q[next_state][next_action] if not done else 0)
            td_error = td_target-Q[state][action]
            Q[state][action]+=alpha*td_error

            state, action = next_state, next_action
        
        epsilon = max(0.01, epsilon*0.995)

        if episode%1000==0:
            print(f"Episode {episode}/{num_episodes} completed")

    return Q

def extract_policy(Q, n_actions):
    """
    Return a greedy policy derived from the learned Q-values.
    """
    def policy_fn(state):
        action_probs = np.zeros(n_actions)
        best_action = np.argmax(Q[state])
        action_probs[best_action] = 1.0
        return action_probs
    return policy_fn

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)

    Q = sarsa(env, num_episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1)
    greedy_policy = extract_policy(Q, env.action_space.n)

    print("\nSample learned Q-values:")
    for s in range(env.observation_space.n):
        print(f"State {s}: {np.round(Q[s], 3)}")

    print("\nGreedy policy (0=←, 1=↓, 2=→, 3=↑):")
    policy_grid = np.array([np.argmax(Q[s]) for s in range(env.observation_space.n)]).reshape(4, 4)
    print(policy_grid)
