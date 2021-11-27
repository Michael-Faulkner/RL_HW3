import time

import numpy as np
import gym
import tensorflow as tf
from collections import deque
from tensorflow.keras import layers
import random
import os
import matplotlib.pyplot as plt


model = tf.keras.models.load_model("pacman_dqn_512_1_final.h5")

episode_rewards = []
env = gym.make('MsPacman-v0')

color = np.array([210, 164, 74]).mean()
epsilon = 0.35
def preprocess_observation(obs):

    # Crop and resize the image
    img = obs[1:176:2, ::2]

    # Convert the image to greyscale
    img = img.mean(axis=2)

    # Improve image contrast
    img[img==color] = 0

    # Next we normalize the image from -1 to +1
    img = (img - 128) / 128 - 1

    return img.reshape(88,80,1)



for j in range(1000):
    episode_reward = 0

    state = env.reset()
    for i in range(20):
        state, reward, done, info = env.step(0)
    state = preprocess_observation(state)
    episode_reward = 0

    q_total = 0
    count = 0
    while True:

        random_number = np.random.uniform()
        #
        # if True:
        #     env.render()
        if random_number <= epsilon:

            action = env.action_space.sample()

        else:
            state_fix = np.array([state])
            q_values = model(state_fix)
            action = np.argmax(q_values)

        next_state, reward, done, info = env.step(action)
        next_state = preprocess_observation(next_state)

        episode_reward += reward

        state = next_state

        if done:
            #print('Episode Number: '+ str(j) + ' Reward ' + str(episode_reward))
            episode_rewards.append(episode_reward)
            break

plt.plot(episode_rewards)
plt.savefig('pacman.jpg')
print(np.mean(episode_rewards))