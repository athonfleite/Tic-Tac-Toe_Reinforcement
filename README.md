# Tic-Tac-Toe Reinforcement Learning Agent

## Project Overview

This project implements a simple reinforcement learning agent using Q-learning to play the game of Tic-Tac-Toe. The goal is to train the agent to play optimally against human players or other agents.

## Project Structure

tic_tac_toe_rl/
├── README.md
├── main.py
├── tic_tac_toe.py
├── q_learning_agent.py
└── utils.py

- `README.md`: This file, containing an overview of the project.
- `main.py`: The main script to train the agent and play the game.
- `tic_tac_toe.py`: Contains the implementation of the Tic-Tac-Toe game environment.
- `q_learning_agent.py`: Contains the Q-learning agent implementation.
- `utils.py`: Contains helper functions, including the training loop.

## How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/tic-tac-toe-rl.git
   cd tic_tac_toe_rl
Install dependencies:
Ensure you have Python 3 installed. You can create a virtual environment and install any dependencies if needed.

Run the main script:

bash
Copiar código
python main.py
The script will train the Q-learning agent for 10,000 episodes and then allow you to play against the trained agent.

How It Works
TicTacToe Environment:

Manages the game state and handles the logic for moves, checking wins, and rendering the board.
reset() initializes a new game.
step(action) executes a move and returns the new state, reward, and whether the game is done.
render() prints the current board state.
QLearningAgent:

Implements the Q-learning algorithm.
get_q(state, action) retrieves the Q-value for a given state-action pair.
update_q(state, action, reward, next_state, done) updates the Q-value based on the reward and the next state's Q-values.
select_action(state) selects an action based on an epsilon-greedy policy.
Training:

The train function runs the training loop for a specified number of episodes, allowing the agent to learn by playing against itself.
Playing Against the Agent:

After training, you can play against the trained agent by running the main script and following the prompts.
Future Improvements
Implement a graphical user interface (GUI) for better interaction.
Experiment with more advanced RL algorithms like Deep Q-learning.
Enhance the agent's strategy with techniques like reward shaping or transfer learning.
