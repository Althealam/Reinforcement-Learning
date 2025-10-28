import numpy as np
import gym

def value_iteration(env, gamma = 0.9, theta = 1e-6):
    """
    Value Iteration algorithm for solving a known MDP
    Args:
        env: OpenAI Gym environment
        gamma: discount factor
        theta: convergence threshold
    Returns:
        V: optimal state-value function
        policy: optimal deterministic policy
    """

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    print("The number of states:", n_states)
    print("The number of actions:", n_actions)

    V = np.zeros(n_states) # initialize value function
    iteration = 0

    # Find the optimal value function 
    while True:
        delta = 0
        for s in range(n_states): # iterate current statement
            # Compute Q(s, a) for all actions
            q_values = np.zeros(n_actions)
            for a in range(n_actions): # iterate all actions
                for prob, next_state, reward, done in env.P[s][a]: # iteratte next state
                    q_values[a]+=prob*(reward+gamma*V[next_state])
            
            # Update V(s)
            best_action_value = np.max(q_values)
            delta = max(delta, np.abs(best_action_value-V[s])) # convergence check
            V[s] = best_action_value
        
        iteration+=1
        if delta<theta:
            print(f"Value Iteration converged after {iteration} iterations.")
            break
    
    # Derive optimal policy from optimal value function
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states): # iterate current statement
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                q_values[a]+=prob*(reward+gamma*V[next_state])
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0
    
    return V, policy


if __name__  == '__main__':
    env = gym.make("FrozenLake-v1", is_slippery= False)
    optimal_values, optimal_policy = value_iteration(env, gamma = 0.9, theta = 1e-6)
    print("Optimal_values:", optimal_values)
    print("Optimal Policy:", optimal_policy)

    # display results
    print("\nOptimal state-value function V*:")
    print(optimal_values.reshape(4, 4))

    action_map = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    policy_grid = np.array([action_map[np.argmax(a)] for a in optimal_policy]).reshape(4, 4)
    print("\nOptimal Policy pi*:")
    print(policy_grid)