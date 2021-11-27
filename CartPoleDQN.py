"""
A Minimal Deep Q-Learning Implementation (minDQN)
Running this code will render the agent solving the CartPole environment using OpenAI gym. Our Minimal Deep Q-Network is approximately 150 lines of code. In addition, this implementation uses Tensorflow and Keras and should generally run in less than 15 minutes.
Usage: python3 minDQN.py
"""

import gym
import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import random

RANDOM_SEED = 24
tf.random.set_seed(RANDOM_SEED)

env = gym.make('CartPole-v0')
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def create_model(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model


def get_qs(model, state):
    return model.predict(state.reshape([1, state.shape[0]]))[0]


def train(replay_memory, model, target_model):
    learning_rate = 0.001  # Learning rate
    discount_factor = 0.95

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


def main():
    epsilon = 1  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1  # You can't explore more than 100% of the time
    min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    model = create_model(env.observation_space.shape, env.action_space.n)
    target_model = create_model(env.observation_space.shape, env.action_space.n)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)
    steps_to_update_target_model = 0
    qs = []
    rewards = deque(maxlen=50)
    episode = 0

    while True:
        observation = env.reset()
        done = False
        episode_reward = 0
        q_total = 0
        count = 0

        while not done:
            steps_to_update_target_model += 1
            random_number = np.random.rand()
            if random_number <= epsilon:
                encoded = observation
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()
                action = env.action_space.sample()
            else:
                encoded = observation
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)

            q_total += np.max(predicted)
            new_observation, reward, done, info = env.step(action)
            replay_memory.append([observation, action, reward, new_observation, done])
            episode_reward += reward
            count += 1
            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(replay_memory, model, target_model)

            observation = new_observation


            if done:

                rewards.append(episode_reward)
                qs.append(q_total/count)
                episode += 1

                if steps_to_update_target_model >= 100:
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        if episode % 50 == 0:
            with open('cartpole.txt', 'w') as f:
                for p in range(len(qs)):
                    f.write(str(qs[p]) + "\n")
            model.save('cartpole.h5')

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    env.close()


if __name__ == '__main__':
    main()