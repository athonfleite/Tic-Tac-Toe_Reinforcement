from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent
from utils import train
from gui import TicTacToeGUI
import tkinter as tk

if __name__ == "__main__":

    root = tk.Tk()
    gui = TicTacToeGUI(root)
    root.mainloop()
