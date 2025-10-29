import gymnasium as gym
import numpy as np
from collections import defaultdict
from collections import deque


def td_n_prediction(env, policy, n=3, num_episodes=5000, alpha=0.1, gamma=0.9):
    """
    n-step TD prediction (for policy evaluation).
    Args:
        env: Gymnasium environment
        policy: function mapping state -> action probabilities
        n: number of steps to look ahead
        num_episodes: number of episodes to run
        alpha: learning rate
        gamma: discount factor
    Returns:
        V: estimated state-value function
    """
    V = defaultdict(float)
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        states = [state]
        rewards = [0]  # reward[0] is placeholder for indexing convenience
        T = float("inf")
        t = 0
        tau = 0

        while True:
            if t < T:
                # Select action following given policy
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1  # terminal time step

            tau = t - n + 1  # time whose estimate is being updated
            if tau >= 0:
                # Compute n-step return G
                G = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]
                if tau + n < T:
                    G += (gamma ** n) * V[states[tau + n]]

                s_tau = states[tau]
                V[s_tau] += alpha * (G - V[s_tau])

            if tau == T - 1:
                break

            t += 1
            state = next_state

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed")

    return V


def create_random_policy(n_actions):
    """Uniform random policy."""
    def policy_fn(state):
        return np.ones(n_actions) / n_actions
    return policy_fn


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    random_policy = create_random_policy(env.action_space.n)

    # Try n = 3 for a middle ground between TD(0) and Monte Carlo
    V = td_n_prediction(env, random_policy, n=3, num_episodes=5000, alpha=0.1, gamma=0.9)

    print("\nLearned state-value function:")
    for s in range(env.observation_space.n):
        print(f"State {s}: {V[s]:.3f}")
