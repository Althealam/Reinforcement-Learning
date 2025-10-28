import gymnasium as gym
import numpy as np
from collections import defaultdict

def td0_prediction(env, policy, num_episodes=10000, alpha=0.1, gamma=0.9):
    """
    TD(0) policy evaluation
    Args:
        env: Gymnasium environment
        policy: function mapping state -> action probabilities
        num_episodes: number of training episodes
        alpha: learning rate
        gamma: discount factor
    Returns:
        V: estimated state-value function
    """
    V = defaultdict(float)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD(0) update rule
            td_target = reward + gamma*V[next_state]
            td_error = td_target-V[state]
            V[state]+=alpha*td_error

            state = next_state
        
        if (episode+1)%1000==0:
            print(f"Episode {episode+1}/{num_episodes} completed")
    return V

def create_random_policy(n_actions):
    """
    Uniform random policy
    """
    def policy_fn(state):
        return np.ones(n_actions)/n_actions
    return policy_fn

if __name__ == '__main__':
    env = gym.make("FrozenLake-v1", is_slippery = False)
    random_policy = create_random_policy(env.action_space.n)
    print("random_policy:", random_policy)

    V = td0_prediction(env, random_policy, num_episodes=5000, alpha = 0.1, gamma = 0.9)

    print("\nLearned state-value function:")
    for s in range(env.observation_space.n):
        print(f"State {s}: {V[s]:.3f}")