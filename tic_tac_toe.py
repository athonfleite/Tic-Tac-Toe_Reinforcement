import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            return self.board.flatten(), -10, True  # Invalid move

        self.board[row, col] = self.current_player
        if self.check_win(self.current_player):
            reward = 1 if self.current_player == 1 else -1
            return self.board.flatten(), reward, True  # Current player wins

        if np.all(self.board != 0):
            return self.board.flatten(), 0, True  # Draw

        self.current_player = -self.current_player
        return self.board.flatten(), 0, False  # Continue game

    def check_win(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == player:
            return True
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == player:
            return True
        return False

    def render(self):
        for row in self.board:
            print(' '.join(['X' if x == 1 else 'O' if x == -1 else '.' for x in row]))
        print()