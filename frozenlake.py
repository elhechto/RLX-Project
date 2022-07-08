import numpy as np
import gym

environment = gym.make('FrozenLake-v1', is_slippery=False)

# Training
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

episodes = 1000
alpha = 0.6
gamma = 0.9

for _ in range(episodes):
    state = environment.reset()
    done = False

    while not done:
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()

        new_state, reward, done, info = environment.step(action)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        state = new_state

print("Training completed!")

# Evaluation
episodes = 100
nb_success = 0

for _ in range(100):
    state = environment.reset()
    done = False

    while not done:
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
        new_state, reward, done, info = environment.step(action)
        state = new_state
        nb_success += reward

print(f"Success rate = {nb_success / episodes * 100}%")

# In Action
print("Let's see the trained agent in action!")

state = environment.reset()
done = False
rewards = 0

for s in range(episodes):
    action = np.argmax(qtable[state,:])
    new_state, reward, done, info = environment.step(action)
    rewards += reward
    environment.render()
    state = new_state
    if done:
        print(f"Finished in {format(s+1)} steps.")
        break

environment.close()
