def train(agent, env, episodes):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            reward = agent.reward_shaping(state, action, reward)  # Apply reward shaping
            agent.update_q(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()