import shutil
import collections
from pathlib import Path
import time
import psutil

import numpy as np
import tensorflow as tf
import gym

from dopamine.dopamine.discrete_domains.atari_lib import create_atari_environment

from model import QuantileQNetwork
from util import preprocess
from buffer import load_dataset


class CQLAgent:
    def __init__(self, env_id, n_atoms=100, gamma=0.99, kappa=1.0, cql_weight=1.0):
        self.env_id = env_id
        self.action_space = gym.make('BreakoutDeterministic-v4').action_space.n

        self.n_atoms = n_atoms
        self.quantiles = \
            [1 / (2 * self.n_atoms) + i * 1 / self.n_atoms for i in range(self.n_atoms)]

        self.qnetwork = QuantileQNetwork(action_space=self.action_space, n_atoms=self.n_atoms)
        self.target_qnetwork = QuantileQNetwork(action_space=self.action_space,
                                                n_atoms=self.n_atoms)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, epsilon=0.00031)

        self.gamma = gamma
        self.kappa = kappa
        self.cql_weight = cql_weight

        self.setup()  # Build qnetwork & target_qnetwork

    def setup(self):
        frames = collections.deque(maxlen=4)

        env = gym.make('BreakoutDeterministic-v4')
        frame = preprocess(env.reset())  # (84,84)

        for _ in range(4):
            frames.append(frame)

        state = np.stack(frames, axis=2)  # (84,84,4)
        state = np.expand_dims(state, axis=0)  # (1,84,84,4)

        self.qnetwork(state)  # build
        self.target_qnetwork(state)

        self.sync_target_weights()

    def save(self, save_dir):
        save_dir = Path(save_dir)

        self.qnetwork.save_weights(str(save_dir / 'qnetwork'))

    def load(self, load_dir="checkpoints/"):
        load_dir = Path(load_dir)

        self.qnetwork.load_weights(str(load_dir / 'qnetwork'))
        self.target_qnetwork.load_weights(str(load_dir / 'qnetwork'))

    def update_network(self, minibatch):
        """
        states, next_states : tf_tensor (48,84,84,4)
        actions, rewards, dones : tf_tensor (48,)
        """

        states, actions, rewards, next_states, dones = minibatch

        # TQ = reward + gamma * Q(s, max_a)
        target_quantile_qvalues = self.make_target_distribution(rewards, next_states, dones)

        with tf.GradientTape() as tape:
            quantile_qvalues_all = self.qnetwork(states)  # (b,action,n_atoms)

            actions_onehot = tf.one_hot(actions, self.action_space, axis=-1)  # (b,action)
            actions_onehot = tf.expand_dims(actions_onehot, axis=-1)  # (b,action,1)

            quantile_qvalues = quantile_qvalues_all * actions_onehot  # (b,actions,n_atoms)
            quantile_qvalues = tf.reduce_sum(quantile_qvalues, axis=1)  # (b,n_atoms)

            td_loss = self.quantile_huberloss(target_quantile_qvalues, quantile_qvalues)  # (b,)

            # CQL(H)
            # log_Z
            Q_learned_all = tf.reduce_mean(quantile_qvalues_all, axis=-1)  # (b,action_dim)
            log_Z = tf.reduce_logsumexp(Q_learned_all, axis=-1)  # (b,)

            # Q_behavior
            Q_behavior = tf.reduce_mean(quantile_qvalues, axis=-1)  # (b,)

            # CQL loss
            cql_loss = log_Z - Q_behavior  # (b,)

            # CQL(H)
            loss = tf.reduce_mean(self.cql_weight * cql_loss + td_loss)

        variables = self.qnetwork.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        # For monitoring
        Q_learned_max = tf.reduce_max(Q_learned_all, axis=1)  # (b,)
        Q_diff = (Q_behavior - Q_learned_max)

        info = {
            'total_loss': loss.numpy(),
            'cql_loss': tf.reduce_mean(cql_loss).numpy(),
            'td_loss': tf.reduce_mean(td_loss).numpy(),
            'rewards': rewards.numpy().sum(),
            'q_behavior': tf.reduce_mean(Q_behavior).numpy(),
            'q_diff': tf.reduce_mean(Q_diff).numpy(),
            'dones': dones.numpy().sum(),
        }

        return info

    def make_target_distribution(self, rewards, next_states, dones):
        next_quantile_qvalues_all = self.target_qnetwork(next_states)  # (b,action,n_atoms)

        next_qvalues_all = tf.reduce_mean(next_quantile_qvalues_all, axis=-1)  # (b,action)

        next_actions = tf.argmax(next_qvalues_all, axis=-1)  # (b,)
        next_actions_onehot = tf.one_hot(next_actions, self.action_space)  # (b,action)
        next_actions_onehot = tf.expand_dims(next_actions_onehot, axis=-1)  # (b,action,1)

        next_quantile_qvalues = tf.reduce_sum(
            next_quantile_qvalues_all * next_actions_onehot,
            axis=1
        )  # (b,n_atoms)

        target_quantile_qvalues = \
            tf.expand_dims(rewards, axis=-1) + \
            self.gamma * (1 - tf.expand_dims(dones, axis=-1)) * next_quantile_qvalues  # (b,n_atoms)

        return target_quantile_qvalues

    @tf.function
    def quantile_huberloss(self, target_quantile_values, quantile_values):
        target_quantile_values = tf.repeat(
            tf.expand_dims(target_quantile_values, axis=1), self.n_atoms, axis=1
        )  # (b,n_atoms,n_atoms)

        quantile_values = tf.repeat(
            tf.expand_dims(quantile_values, axis=2), self.n_atoms, axis=2
        )  # (b,n_atoms,n_atoms)

        errors = target_quantile_values - quantile_values  # (b,n_atoms,n_atoms)

        is_smaller_than_kappa = tf.abs(errors) < self.kappa  # (b,n_atoms,n_atoms)

        squared_loss = .5 * tf.square(errors)  # (b,n_atoms,n_atoms)
        linear_loss = self.kappa * (tf.abs(errors) - 0.5 * self.kappa)  # (b,n_atoms,n_atoms)

        huber_loss = \
            tf.where(is_smaller_than_kappa, squared_loss, linear_loss)  # (b,n_atoms,n_atoms)

        indicator = tf.stop_gradient(tf.where(errors < 0, 1., 0.))  # (b,n_atoms,n_atoms)
        quantiles = \
            tf.repeat(
                tf.expand_dims(self.quantiles, axis=1), self.n_atoms, axis=1
            )  # (n_atoms,n_atoms)

        quantile_weights = tf.abs(quantiles - indicator)  # (b,n_atoms,n_atoms)

        quantile_huber_loss = quantile_weights * huber_loss  # (b,n_atoms,n_atoms)

        td_loss = tf.reduce_mean(quantile_huber_loss, axis=-1)  # (b,n_atoms)
        td_loss = tf.reduce_sum(td_loss, axis=1)  # (b,)

        return td_loss

    def sync_target_weights(self):
        weights = self.qnetwork.get_weights()
        self.target_qnetwork.set_weights(weights)

    def rollout(self):

        env = create_atari_environment(game_name=self.env_id, sticky_actions=True)

        rewards, steps = 0, 0

        frames = collections.deque(maxlen=4)
        for _ in range(4):
            frames.append(np.zeros((84, 84), dtype=np.float32))

        frame = env.reset()[:, :, 0]  # env.reset: (84,84,1)
        frames.append(frame)

        done = False

        while (not done) and (steps < 3000):
            state = np.stack(frames, axis=2)
            state = np.expand_dims(state, axis=0)

            action = self.qnetwork.sample_action(state)

            next_frame, reward, done, _ = env.step(action)

            frames.append(next_frame[:, :, 0])  # next_frames: (84,84,1)

            rewards += reward
            steps += 1

        return rewards, steps


def train(n_iter=20000000,
          env_id='BreakoutDeterministic-v4',
          dataset_dir='tfrecords_dqn_replay_dataset/',
          batch_size=48,
          cql_weight=4.0,
          target_update_period=8000,
          resume_from=None):
    logdir = Path(__file__).parent / 'log'
    if logdir.exists() and resume_from is None:
        shutil.rmtree(logdir)

    summary_writer = tf.summary.create_file_writer(str(logdir))

    agent = CQLAgent(env_id=env_id, cql_weight=cql_weight)

    # dataset that reads tfrecord files, then make minibatch
    dataset = load_dataset(dataset_dir=dataset_dir, batch_size=batch_size)

    if resume_from is not None:
        agent.load()
        n = int(resume_from * 1000)
    else:
        n = 1

    s = time.time()

    for minibatch in dataset:
        info = agent.update_network(minibatch)

        if n % 25 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('loss', info['total_loss'], step=n)
                tf.summary.scalar('cql_loss', info['cql_loss'], step=n)
                tf.summary.scalar('td_loss', info['td_loss'], step=n)
                tf.summary.scalar('q_behavior', info['q_behavior'], step=n)
                tf.summary.scalar('q_diff', info['q_diff'], step=n)
                tf.summary.scalar('dones', info['dones'], step=n)

        if n % target_update_period == 0:
            agent.sync_target_weights()

        if n % 2500 == 0:  # default=2500
            rewards, steps = agent.rollout()
            mem = psutil.virtual_memory().used / (1024 ** 3)

            with summary_writer.as_default():
                tf.summary.scalar('test_score', rewards, step=n)
                tf.summary.scalar('test_steps', steps, step=n)
                tf.summary.scalar('laptime', time.time() - s, step=n)
                tf.summary.scalar('Mem', mem, step=n)

            s = time.time()
            print(f'=== test: {n} ===')
            print(f'score: {rewards}, steps:{steps}')

        if n % 25000 == 0:  # defaule=25000
            agent.save(save_dir='checkpoints')

        if n > n_iter:
            break

        n += 1


def main():
    env_id = 'Breakout'
    dataset_dir = 'tfrecords_dqn_replay_dataset/'

    train(env_id=env_id, cql_weight=4.0, resume_from=None, dataset_dir=dataset_dir)


if __name__ == '__main__':
    main()
