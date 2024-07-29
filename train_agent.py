from q_learning_agent import QLearningAgent
from tic_tac_toe import TicTacToe
from utils import train
from utils import save_model

# Initialize the environment and agent
env = TicTacToe()
agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_min=0.000000000000000001, epsilon_decay=0.999, buffer_size=2000)

# Train the agent
train(agent, env, episodes=100000, verbose=True)

# Save the trained model
save_model(agent, 'q_learning_model.pkl')