import random
import numpy as np
from collections import deque

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=2000):
        self.q_table = {}  # Initialize Q-table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer = deque(maxlen=buffer_size)  # Experience replay buffer
        self.batch_size = 64

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

    def store_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_experiences(self):
        if len(self.buffer) < self.batch_size:
            return []
        return random.sample(self.buffer, self.batch_size)

    def learn_from_experiences(self):
        experiences = self.sample_experiences()
        if not experiences:
            return
        states, actions, rewards, next_states, dones = zip(*experiences)

        for i in range(self.batch_size):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]

            best_next_q = max([self.get_q(next_state, a) for a in range(9) if next_state[a] == 0], default=0)
            current_q = self.get_q(state, action)
            target = reward if done else reward + self.gamma * best_next_q
            self.q_table[(tuple(state), action)] = current_q + self.alpha * (target - current_q)