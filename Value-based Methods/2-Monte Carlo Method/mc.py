import gymnasium as gym
import numpy as np
from collections import defaultdict

def create_random_policy(n_actions):
    """
    Return a policy that chooses each action with equal probability
    """
    def policy_fn(observation):
        return np.ones(n_actions) / n_actions
    return policy_fn

def create_greedy_policy(Q):
    """
    Return a greedy policy derived from Q(s, a)
    """
    def policy_fn(observation):
        n_actions = len(Q[observation])
        action_probabilities = np.zeros(n_actions)
        best_action = np.argmax(Q[observation])
        action_probabilities[best_action] = 1.0
        return action_probabilities
    return policy_fn

def monte_carlo_control(env, num_episodes, gamma=1.0, epsilon = 0.1):
    """
    Monte Carlo control using epsilon-soft exploring starts
    Args:
        env: OpenAI Gym environment 
        num_episodes: number of training episodes
        gamma: discount factor
        epsilon: exploration probability for epsilon-soft policy
    Returns:
        Q: action-value function
        policy: derived greedy policy
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # Tabular Q: maps each state(a tuple) to a vector of actions values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for episode in range(1, num_episodes+1):
        # Generate one episode: list of (state, action, reward)
        episode_data = [] # store the trajectory (s_t, a_t, r_t+1)
        state = env.reset()[0]
        done = False

        while not done:
            # epsilon action selection
            if np.random.rand()<epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_data.append((state, action, reward))
            state = next_state
        
        # Compute returns G_t for each (state, action)
        visited_state_actions = set()
        G = 0.0
        for (state, action, reward) in reversed(episode_data):
            G = gamma*G+reward
            if (state, action) not in visited_state_actions:
                returns_sum[(state, action)]+=G
                returns_count[(state, action)]+=1.0
                # returns_sum/returns_count: store cumulative returns and counts per (state, action) to compute the running mean
                Q[state][action] = returns_sum[(state, action)]/returns_count[(state, action)]
                visited_state_actions.add((state, action))
            
            # Logging progress
            if episode%10000==0:
                print(f"Episode {episode}/{num_episodes} completed.")
        
        # Derive greedy policy from final Q
        def policy_fn(observation):
            """
            returns the greedy one-hot policy derived from learned Q
            """
            action_probs = np.zeros(env.action_space.n)
            best_action = np.argmax(Q[observation])
            action_probs[best_action] = 1.0
            return action_probs
    return Q, policy_fn

if __name__ == '__main__':
    env = gym.make("Blackjack-v1")

    Q, policy = monte_carlo_control(env, num_episodes=500000, gamma=1.0, epsilon=0.1)
    # Evaluate final policy on a few random states
    print("\nSample learned Q-values:")
    for state, actions in list(Q.items())[:5]:
        print(f"State: {state}, Q:{np.round(actions, 3)}")
    
    # Demonstrate greedy policy action
    test_state = (20, 10, False) # player_sum = 20, dealer card = 10, no usable act
    best_action = np.argmax(Q[test_state])
    print(f"\nFor state {test_state}, best action = {best_action} (0=stick, 1=hit)")
