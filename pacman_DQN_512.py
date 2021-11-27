import numpy as np
import gym
import tensorflow as tf
from collections import deque
from tensorflow.keras import layers
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "6"

def DQN():
    inputs = tf.keras.Input(shape = (88,80,1))
    x1 = layers.Conv2D(32, 8, strides = (4, 4), padding = 'same', activation='relu')(inputs)
    x2 = layers.Conv2D(64, 4, strides=(2, 2), padding='same', activation='relu')(x1)
    x3 = layers.Conv2D(64, 3, padding='same', activation='relu')(x2)
    x4 = layers.Flatten()(x3)
    x5 = layers.Dense(512, activation='relu')(x4)
    output = layers.Dense(9, activation='linear')(x5)

    model = tf.keras.Model(inputs, output)
    return model

color = np.array([210, 164, 74]).mean()

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

def update_base(base_model, target_model, replay_memory):
    batch_size = 64
    mini_batch = random.sample(replay_memory, batch_size)

    states = np.array([x[0] for x in mini_batch])
    next_states = [x[3] for x in mini_batch]
    actions = [x[1] for x in mini_batch]
    rewards = [x[2] for x in mini_batch]
    dones = [x[4] for x in mini_batch]

    Qs = base_model.predict(states)

    Y = []
    for i in range(len(mini_batch)):
        if not dones[i]:
            next_state_fix = np.array([next_states[i]])
            future_Q = target_model(next_state_fix)
            max_future_Q = rewards[i] + 0.99 * np.max(future_Q)
        else:
            max_future_Q = rewards[i]

        Qs[actions[i]] = (1 - 0.5) * Qs[actions[i]] + 0.5 * max_future_Q

        Y.append(Qs)
    y = np.array(Y)
    base_model.fit(x=states, y=y, verbose=0, batch_size=64)



def main():

    env = gym.make('MsPacman-v0')

    epsilon = 1  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1  # You can't explore more than 100% of the time
    min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    base_model = DQN()
    target_model = DQN()

    base_model.compile(loss = tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0025), metrics = 'mse' )
    target_model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0025), metrics = 'mse')
    target_model.set_weights(base_model.get_weights())

    replay_memory = deque(maxlen=10000)

    steps = 0
    episode = 0
    burnin = 2000
    q_list = []
    while True:
        state = env.reset()
        for i in range(20):
            state, reward, done, info = env.step(0)
        state = preprocess_observation(state)
        episode_reward = 0

        q_total = 0
        count = 0
        while True:

            random_number = np.random.uniform()

            if random_number <= epsilon:
                state_fix = np.array([state])
                action = env.action_space.sample()
                q_values = base_model(state_fix)

            else:
                state_fix = np.array([state])
                q_values = base_model(state_fix)
                action = np.argmax(q_values)

            q_total += np.max(q_values)
            count += 1

            next_state, reward, done, info = env.step(action)
            next_state = preprocess_observation(next_state)
            if reward == 0:
                reward = -0.1
            episode_reward += reward
            replay_memory.append((state, action, reward, next_state, done))

            if steps % 4 == 0 and burnin < steps:
                update_base(base_model, target_model, replay_memory)

            if steps % 1000 == 0:
                target_model.set_weights(base_model.get_weights())

            steps += 1
            state = next_state

            if done:
                print('Episode Number: '+ str(episode) + ' Reward ' + str(episode_reward))
                episode += 1
                q_list.append(q_total/count)
                break

        if episode % 50 == 0:
            with open("pacman_q_512.txt" 'w') as f:
                f.writelines(q_list)

            base_model.save('pacman_dqn_512.h5')

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

main()