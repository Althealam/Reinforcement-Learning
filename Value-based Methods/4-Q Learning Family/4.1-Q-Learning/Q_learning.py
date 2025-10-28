from math import trunc
import gymnasium as gym
import numpy as np
from collections import defaultdict

def q_learning(env, num_episodes=5000, alpha = 0.1, gamma = 0.9, epsilon=0.1):
    """
    Q-Learning algorithm (off policy TD controls)
    Args:
        env: Gymnasium environment
        num_episodes: number of training episodes
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate for epsilon-greedy behavior policy
    Returns:
        Q: action-value table
    """
    Q  = defaultdict(lambda: np.zeros(env.action_space.n))

    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        done = False

        while not done:
            # epsilon-greedy action selection
            if np.random.rand()<epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-Learning update (off-policy TD)
            best_next_action = np.argmax(Q[next_state])
            td_target = reward+gamma*Q[next_state][best_next_action]
            td_error = td_target-Q[state][action]
            Q[state][action]+=alpha*td_error

            state = next_state
        
        if episode%1000==0:
            print(f"Episode {episode}/{num_episodes} completed")
    
    return Q

def extract_policy(Q, n_actions):
    """
    Derive a greedy policy from the learning Q-values
    """
    def policy_fn(state):
        action_probs = np.zeros(n_actions)
        best_action = np.argmax(Q[state])
        action_probs[best_action] = 1.0
        return action_probs
    return policy_fn

if __name__ == '__main__':
    # Deterministic FrozenLake (for stability)
    env = gym.make("FrozenLake-v1", is_slippery=False)

    Q = q_learning(env, num_episodes=5000, alpha = 0.1, gamma = 0.9, epsilon=0.9)
    greedy_policy = extract_policy(Q, env.action_space.n)

    print("\nSample learned Q-values:")
    for s in range(env.observation_space.n):
        print(f"State {s}: {np.round(Q[s], 3)}") # Q(s, a) for different action and statement(4 actions)
    
    print("\nGreedy policy (0=←, 1=↓, 2=→, 3=↑):")
    policy_grid = np.array([np.argmax(Q[s]) for s in range(env.observation_space.n)]).reshape(4, 4)
    print(policy_grid)