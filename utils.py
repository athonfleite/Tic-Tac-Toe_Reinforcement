import pickle
import time

def save_model(agent, filename):
    with open(filename, 'wb') as f:
        pickle.dump(agent.q_table, f)

def load_model(agent, filename):
    with open(filename, 'rb') as f:
        agent.q_table = pickle.load(f)

def train(agent, env, episodes, verbose=True):
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        start_time = time.time()
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            reward = agent.reward_shaping(state, action, reward)
            agent.store_experience(state, action, reward, next_state, done)
            agent.learn_from_experiences()
            state = next_state
            episode_reward += reward
        
        agent.decay_epsilon()
        
        if verbose and (episode + 1) % 100 == 0:  # Print every 100 episodes
            elapsed_time = time.time() - start_time
            print(f"Episode {episode + 1}/{episodes} | Reward: {episode_reward} | Epsilon: {agent.epsilon:.4f} | Time: {elapsed_time:.2f}s")