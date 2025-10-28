import numpy as np
import gym


def policy_evaluation(policy, env, gamma=0.9, theta=1e-6):
    """Evaluate a policy given an environment and a full definition of policy."""
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def policy_improvement(V, env, gamma=0.9):
    """Greedy policy improvement step."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    new_policy = np.zeros((n_states, n_actions))

    for s in range(n_states): # iterate current statement
        action_values = np.zeros(n_actions)
        for a in range(n_actions): # iterate actions
            for prob, next_state, reward, done in env.P[s][a]: # iterate next statement
                action_values[a] += prob * (reward + gamma * V[next_state])
        best_action = np.argmax(action_values) # update best action
        new_policy[s][best_action] = 1.0 # update policy
    return new_policy


def policy_iteration(env, gamma=0.9, theta=1e-6):
    """Policy Iteration algorithm: alternate between evaluation and improvement."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Start with a uniform random policy
    policy = np.ones((n_states, n_actions)) / n_actions
    iteration = 0

    while True:
        iteration += 1
        V = policy_evaluation(policy, env, gamma, theta) 
        new_policy = policy_improvement(V, env, gamma)

        # Stop if policy did not change
        if np.all(policy == new_policy):
            print(f"✅ Policy Iteration converged after {iteration} iterations.")
            break
        policy = new_policy

    return policy, V


if __name__ == "__main__":
    # Create environment
    env = gym.make("FrozenLake-v1", is_slippery=False)

    optimal_policy, optimal_value = policy_iteration(env, gamma=0.9, theta=1e-6)

    print("Optimal state-value function V*:")
    print(optimal_value.reshape(4, 4))

    # Map numeric actions to directions
    action_map = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    policy_grid = np.array([action_map[np.argmax(a)] for a in optimal_policy]).reshape(4, 4)
    print("Optimal policy π*:")
    print(policy_grid)
