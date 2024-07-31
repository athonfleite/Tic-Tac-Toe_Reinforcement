from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent
from utils import train, load_model, save_model
from gui import TicTacToeGUI
import tkinter as tk

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