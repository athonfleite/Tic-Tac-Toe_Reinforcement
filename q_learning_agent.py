import random
import numpy as np

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get((tuple(state), action), 0)

    def update_q(self, state, action, reward, next_state, done):
        best_next_q = max([self.get_q(next_state, a) for a in range(9) if next_state[a] == 0], default=0)
        current_q = self.get_q(state, action)
        target = reward if done else reward + self.gamma * best_next_q
        self.q_table[(tuple(state), action)] = current_q + self.alpha * (target - current_q)

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([i for i in range(9) if state[i] == 0])
        q_values = [self.get_q(state, a) if state[a] == 0 else -float('inf') for a in range(9)]
        return np.argmax(q_values)