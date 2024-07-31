import random
import numpy as np
from collections import deque

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.1, buffer_size=1000):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer = deque(maxlen=buffer_size)  # Experience replay buffer
        self.batch_size = 64
        self.q_table = {}

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.available_actions(state))
        else:
            action = self.best_action(state)
        return action

    def best_action(self, state):
        state_str = str(state)
        if state_str not in self.q_table or len(self.q_table[state_str]) == 0:
            return random.choice(self.available_actions(state))
        return max(self.q_table[state_str], key=self.q_table[state_str].get)

    def update_q_value(self, state, action, reward, next_state):
        state_str = str(state)
        next_state_str = str(next_state)
        best_next_action = self.best_action(next_state)
        td_target = reward + self.gamma * self.q_table.get(next_state_str, {}).get(best_next_action, 0.0)
        td_error = td_target - self.q_table.setdefault(state_str, {}).get(action, 0.0)
        self.q_table[state_str][action] = self.q_table[state_str][action] + self.alpha * td_error
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def available_actions(self, state):
        return [i for i in range(len(state)) if state[i] == 0]

    def get_q(self, state, action):
        return self.q_table.get((tuple(state), action), 0)

    def update_q(self, state, action, reward, next_state, done):
        best_next_q = max([self.get_q(next_state, a) for a in range(9) if next_state[a] == 0], default=0)
        current_q = self.get_q(state, action)
        target = reward if done else reward + self.gamma * best_next_q
        self.q_table[(tuple(state), action)] = current_q + self.alpha * (target - current_q)


    def select_strategic_action(self, state):
        # Prioritize center and corners if available
        strategic_positions = [4, 0, 2, 6, 8]
        available_positions = [pos for pos in strategic_positions if state[pos] == 0]
        if available_positions:
            return random.choice(available_positions)
        return random.choice([i for i in range(9) if state[i] == 0])

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

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reward_shaping(self, state, action, reward):
        # Example reward shaping for Tic-Tac-Toe
        next_state = state[:]
        next_state[action] = 1  # Assume agent's move is 1

        if self.check_winner(next_state, 1):  # If agent wins
            return 10  # Reward for winning
        elif self.check_winner(next_state, -1):  # If opponent wins
            return -10  # Penalty for losing
        elif 0 not in next_state:  # If the board is full and it's a draw
            return 5  # Reward for draw
        else:
            # Encourage moves that lead to a potentially winning state
            if self.is_potential_win(next_state, 1):  # Check if the move leads to a potential win
                return 1  # Small reward for potential win
            elif self.is_potential_win(next_state, -1):  # Check if the move prevents opponent from winning
                return 1  # Small reward for blocking opponent
            return reward  # Default reward if no special conditions are met

    def is_potential_win(self, state, player):
        # Check if a move leads to a potential win
        for action in range(9):
            if state[action] == 0:
                temp_state = state[:]
                temp_state[action] = player
                if self.check_winner(temp_state, player):
                    return True
        return False

    def check_winner(self, state, player):
        # Check rows, columns, and diagonals for a win
        win_states = [
            [state[0], state[1], state[2]],
            [state[3], state[4], state[5]],
            [state[6], state[7], state[8]],
            [state[0], state[3], state[6]],
            [state[1], state[4], state[7]],
            [state[2], state[5], state[8]],
            [state[0], state[4], state[8]],
            [state[2], state[4], state[6]],
        ]
        return [player, player, player] in win_states