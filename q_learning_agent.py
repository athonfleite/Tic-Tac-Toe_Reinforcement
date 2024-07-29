import random
import numpy as np

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, tau=1.0):
        self.q_table = {}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau  # Temperature parameter for Boltzmann exploration

    def get_q(self, state, action):
        return self.q_table.get((tuple(state), action), 0)

    def update_q(self, state, action, reward, next_state, done):
        best_next_q = max([self.get_q(next_state, a) for a in range(9) if next_state[a] == 0], default=0)
        current_q = self.get_q(state, action)
        target = reward if done else reward + self.gamma * best_next_q
        self.q_table[(tuple(state), action)] = current_q + self.alpha * (target - current_q)

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.boltzmann_action(state)
        q_values = [self.get_q(state, a) if state[a] == 0 else -float('inf') for a in range(9)]
        return np.argmax(q_values)

    def boltzmann_action(self, state):
        q_values = np.array([self.get_q(state, a) if state[a] == 0 else -float('inf') for a in range(9)])
        exp_q = np.exp(q_values / self.tau)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(np.arange(9), p=probs)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reward_shaping(self, state, action, reward):
        # Example reward shaping: encourage winning and discourage losing
        return reward + (0.1 if state[action] == 0 else -0.1)