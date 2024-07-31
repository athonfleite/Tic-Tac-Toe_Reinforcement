import tkinter as tk
from tkinter import messagebox
from q_learning_agent import QLearningAgent
from utils import save_model, load_model
from tic_tac_toe import TicTacToe
from utils import train

class TicTacToeGUI:
    def __init__(self, root, agent):
        self.root = root
        self.agent = agent
        self.board = [0] * 9  # 0 = empty, 1 = player, -1 = agent
        self.buttons = []
        self.create_widgets()
        self.agent_first_move()

    def create_widgets(self):
        for i in range(9):
            button = tk.Button(self.root, text='', font=('normal', 20, 'normal'), width=5, height=2,
                               command=lambda i=i: self.player_move(i))
            button.grid(row=i//3, column=i%3)
            self.buttons.append(button)

    def player_move(self, index):
        if self.board[index] == 0:
            self.board[index] = 1
            self.update_button(index, 'X')
            if self.check_winner(self.board, 1):
                self.end_game("You win!")
            elif 0 not in self.board:
                self.end_game("It's a draw!")
            else:
                self.agent_move()

    def agent_first_move(self):
        action = self.agent.select_action(self.board)
        self.board[action] = -1
        self.update_button(action, 'O')

    def agent_move(self):
        action = self.agent.select_action(self.board)
        self.board[action] = -1
        self.update_button(action, 'O')
        if self.check_winner(self.board, -1):
            self.end_game("You lose!")
        elif 0 not in self.board:
            self.end_game("It's a draw!")

    def update_button(self, index, text):
        self.buttons[index].config(text=text, state=tk.DISABLED)

    def check_winner(self, board, player):
        win_states = [
            [board[0], board[1], board[2]],
            [board[3], board[4], board[5]],
            [board[6], board[7], board[8]],
            [board[0], board[3], board[6]],
            [board[1], board[4], board[7]],
            [board[2], board[5], board[8]],
            [board[0], board[4], board[8]],
            [board[2], board[4], board[6]],
        ]
        return [player, player, player] in win_states

    def end_game(self, message):
        messagebox.showinfo("Game Over", message)
        self.root.destroy()

if __name__ == "__main__":
    agent = QLearningAgent()  # Initialize the agent first
    try:
        load_model(agent, 'q_learning_model.pkl')
    except (FileNotFoundError, EOFError):
        env = TicTacToe()
        train(agent, env, episodes=10000, verbose=True)
        save_model(agent, 'q_learning_model.pkl')
    
    root = tk.Tk()
    root.title("Tic Tac Toe")
    game = TicTacToeGUI(root, agent)
    root.mainloop()