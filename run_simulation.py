import gymnasium as gym
import avoid_bullets_env  # Ortamı kaydettiğimiz dosyayı ithal edin
import numpy as np
import random
import time

# Gymnasium ortamını başlatın
env = gym.make('AvoidBullets-v1')

# Q-table initialization
q_table = np.zeros([6, env.action_space.n])

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration-exploitation tradeoff

# Training
num_episodes = 1000

for i in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state[0]])  # Exploit learned values

        next_state, reward, done, _ = env.step(action)

        old_value = q_table[state[0], action]
        next_max = np.max(q_table[next_state[0]])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state[0], action] = new_value

        state = next_state

print("Training finished.\n")

# Test the agent
for i in range(100):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_table[state[0]])
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.5)  # Her adım arasında kısa bir bekleme süresi ekleyin

    print(f"Episode {i+1}: Total reward: {total_reward}")

env.close()
