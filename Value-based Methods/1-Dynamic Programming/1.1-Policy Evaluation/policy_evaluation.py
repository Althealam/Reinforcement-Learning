import numpy as np
import gym

# Create a deterministic FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Get number of states and actions
n_states = env.observation_space.n
print("The number of states:", n_states)
n_actions = env.action_space.n
print("The number of actions:", n_actions)

# Hyperparameters
gamma = 0.9
theta = 1e-6  # convergence threshold

# Initialize state-value function
V = np.zeros(n_states)

# Define a random policy π(a|s)
policy = np.ones((n_states, n_actions)) / n_actions


def policy_evaluation(policy, env, gamma=0.9, theta=1e-6):
    """
    Evaluate a given policy using iterative policy evaluation.
    Args:
        policy: [n_states, n_actions] array representing π(a|s)
        env: OpenAI Gym environment
        gamma: discount factor
        theta: convergence threshold
    Returns:
        V: estimated state-value function for the given policy
    """
    V = np.zeros(env.observation_space.n)
    iteration = 0

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]): # iterate action with a given policy
                for prob, next_state, reward, done in env.P[s][a]: # iterate next state
                    v += action_prob * prob * (reward + gamma * V[next_state]) # Bellman Equation
            delta = max(delta, abs(v - V[s])) 
            V[s] = v
        iteration += 1
        if delta < theta:
            break

    print(f"✅ Policy Evaluation converged after {iteration} iterations")
    return V


# Run policy evaluation
V_pi = policy_evaluation(policy, env, gamma, theta)

# Print results
print("State-value function Vπ:")
print(V_pi.reshape(4, 4))
