import tkinter as tk
from tkinter import messagebox
from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent
from utils import train

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe with RL Agent")
        self.env = TicTacToe()
        self.agent = QLearningAgent(alpha=0.5, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)
        self.state = self.env.reset()
        self.buttons = [[None for _ in range(3)] for _ in range(3)]

    def create_board(self):
        for i in range(3):
            for j in range(3):
                button = tk.Button(self.root, text='', font='Arial 20', width=5, height=2,
                                   command=lambda i=i, j=j: self.human_move(i, j))
                button.grid(row=i, column=j)
                self.buttons[i][j] = button

    def train_agent(self):
        train(self.agent, self.env, 500000)  # Increase number of training episodes
        self.reset_board()

    def reset_board(self):
        self.state = self.env.reset()
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text='', state=tk.NORMAL)

    def human_move(self, row, col):
        action = row * 3 + col
        if self.env.board[row, col] == 0:
            self.state, reward, done = self.env.step(action)
            self.update_board()
            if done:
                self.end_game(reward)
            else:
                self.agent_move()

    def agent_move(self):
        action = self.agent.select_action(self.state)
        self.state, reward, done = self.env.step(action)
        self.update_board()
        if done:
            self.end_game(reward)

    def update_board(self):
        for i in range(3):
            for j in range(3):
                if self.env.board[i, j] == 1:
                    self.buttons[i][j].config(text='X')
                elif self.env.board[i, j] == -1:
                    self.buttons[i][j].config(text='O')

    def end_game(self, reward):
        if reward == 1:
            message = "You win!"
        elif reward == -1:
            message = "You lose!"
        elif reward == -10:
            message = "Invalid move! You lose!"
        else:
            message = "It's a draw!"
        messagebox.showinfo("Game Over", message)
        self.reset_board()

if __name__ == "__main__":
    root = tk.Tk()
    gui = TicTacToeGUI(root)
    root.mainloop()