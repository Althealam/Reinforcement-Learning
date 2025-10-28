import gymnasium as gym
import numpy as np
from collections import defaultdict

def expected_sarsa(env, num_episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Expected SARSA algorithm (on-policy TD control with expectation).
    Args:
        env: Gymnasium environment
        num_episodes: number of episodes
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate
    Returns:
        Q: learned action-value function
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        done = False

        while not done:
            # epsilon-greedy behavior policy
            if np.random.rand()<epsilon: # explore
                action = env.action_space.sample()
            else: # greedy: choose the action with max Q value (in the cost scene, we need to choose the action with min Q value)
                action = np.argmax(Q[state]) # if there are not only one action which have max value function, we can choose min index
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Compute expected value under epsilon-greedy policy
            action_probs = np.ones(env.action_space.n) * (epsilon / env.action_space.n) # explore probability
            best_action = np.argmax(Q[next_state]) 
            action_probs[best_action] +=(1.0-epsilon) # greedy probability

            expected_value = np.dot(Q[next_state], action_probs) # in the normal SASRA, this place is the Q value for next statement and next action(already decided in the previous step)
            # in the normal Q-learning, this place is the Q value for next statement with max Q value 

            # TD target and update
            td_target = reward+gamma*expected_value
            td_error = td_target - Q[state][action]
            Q[state][action]+=alpha*td_error

            state = next_state
        
        if episode%1000==0:
            print(f"Episode {episode}/{num_episodes} completed")
    
    return Q

def extract_policy(Q, n_actions):
    """
    Return a greedy policy derived from Q
    """
    def policy_fn(state):
        action_probs = np.zeros(n_actions)
        best_action = np.argmax(Q[state])
        action_probs[best_action]=1.0
        return action_probs
    return policy_fn

if __name__=='__main__':
    env = gym.make("FrozenLake-v1", is_slippery = False)
    Q = expected_sarsa(env, num_episodes=5000, alpha=0.1, gamma = 0.9, epsilon=0.1)
    greedy_policy = extract_policy(Q, env.action_space.n)

    print("\nSample learned Q-values:")
    for s in range(env.observation_space.n):
        print(f"State {s}: {np.round(Q[s], 3)}")

    print("\nGreedy policy (0=←, 1=↓, 2=→, 3=↑):")
    policy_grid = np.array([np.argmax(Q[s]) for s in range(env.observation_space.n)]).reshape(4, 4)
    print(policy_grid)
