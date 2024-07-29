from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent
from utils import train

if __name__ == "__main__":
    env = TicTacToe()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)

    # Train the agent
    train(agent, env, 10000)

    # Play a game against the trained agent
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        env.render()
        state = next_state
        if done:
            break

        # Human move
        while True:
            human_action = int(input("Enter your move (0-8): "))
            row, col = divmod(human_action, 3)
            if env.board[row, col] == 0:
                break
            print("Invalid move. Try again.")
        next_state, reward, done = env.step(human_action)
        env.render()
        state = next_state