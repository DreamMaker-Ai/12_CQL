import random

import numpy as np
import tensorflow as tf


class QuantileQNetwork(tf.keras.models.Model):
    def __init__(self, action_space, n_atoms=200):
        super(QuantileQNetwork, self).__init__()

        self.action_space = action_space
        self.n_atoms = n_atoms

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4,
                                            activation='relu', kernel_initializer='he_normal')

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,
                                            activation='relu', kernel_initializer='he_normal')

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                            activation='relu', kernel_initializer='he_normal')

        self.flatten1 = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(units=512, activation='relu',
                                            kernel_initializer='he_normal')

        self.out = tf.keras.layers.Dense(units=self.action_space * self.n_atoms,
                                         kernel_initializer='he_normal')

        self.quantile_values = tf.keras.layers.Reshape(
            target_shape=(self.action_space, self.n_atoms)
        )

    def call(self, x):  # x; (1,84,84,4)

        x = x / 255  # (1,84,84,4)

        x = self.conv1(x)  # (1,20,20,32)
        x = self.conv2(x)  # (1,9,9,64)
        x = self.conv3(x)  # (1,7,7,64)

        x = self.flatten1(x)  # (1,3136)
        out = self.out(x)  # (1,40)

        quantile_values = self.quantile_values(out)  # (1,4,10)

        return quantile_values

    def sample_action(self, state, eps=0.0):  # state: (1,84,84,4)

        assert state.shape[0] == 1

        if random.random() > eps:
            state = tf.cast(state, dtype=tf.float32)
            quantile_qvalues = self(state)  # (1,4,10)

            q_means = tf.reduce_mean(quantile_qvalues, axis=-1, keepdims=False)  # (1,4)

            selected_action = tf.argmax(q_means, axis=-1)  # (1,)
            selected_action = int(selected_action.numpy()[0])  # int

        else:
            selected_action = np.random.choice(self.action_space)

        return selected_action


def main():
    import gym
    import numpy as np
    import collections
    from util import preprocess

    env = gym.make('BreakoutDeterministic-v4')
    frames = collections.deque(maxlen=4)

    frame = preprocess(env.reset())  # (84,84)

    for _ in range(4):
        frames.append(frame)

    state = np.stack(frames, axis=2)  # (84,84,4)
    state = np.expand_dims(state, axis=0)  # (1,84,84,4)

    qnet = QuntileQNetwork(action_space=4, n_atoms=10)

    qnet(state)  # (1,4,10)

    selected_action = qnet.sample_action(state)

    print(selected_action)


if __name__ == "__main__":
    main()
