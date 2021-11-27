import time

import numpy as np
import gym
import tensorflow as tf
from collections import deque
from tensorflow.keras import layers
import random
import os
import matplotlib.pyplot as plt


model = tf.keras.models.load_model("cartpole.h5")

episode_rewards = []
env = gym.make('CartPole-v0')

epsilon = 0.05



for j in range(1000):
    episode_reward = 0

    state = env.reset()

    episode_reward = 0


    while True:

        random_number = np.random.uniform()

        # if True:
        #     env.render()
        #     time.sleep(0.01666667)
        # if random_number <= epsilon:
        #
        #     action = env.action_space.sample()
        #
        # else:
        state_fix = np.array([state])
        q_values = model(state_fix)
        action = np.argmax(q_values)

        next_state, reward, done, info = env.step(action)


        episode_reward += reward

        state = next_state

        if done:
            print('Episode Number: '+ str(j) + ' Reward ' + str(episode_reward))
            episode_rewards.append(episode_reward)
            break

plt.plot(episode_rewards)
plt.savefig('cartpole_test.jpg')
print(np.mean(episode_rewards))