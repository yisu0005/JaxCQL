from collections import defaultdict
from copy import copy, deepcopy
from queue import Queue
import threading

import d4rl

import numpy as np
import jax.numpy as jnp


class ReplayBuffer(object):
    def __init__(self, max_size, data=None):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

        if data is not None:
            if self._max_size < data['observations'].shape[0]:
                self._max_size = data['observations'].shape[0]
            self.add_batch(data)

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.add_sample(o, a, r, no, d)

    def add_batch(self, batch):
        self.add_traj(
            batch['observations'], batch['actions'], batch['rewards'],
            batch['next_observations'], batch['dones']
        )

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices):
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
        )

    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def data(self):
        return dict(
            observations=self._observations[:self._size, ...],
            actions=self._actions[:self._size, ...],
            rewards=self._rewards[:self._size, ...],
            next_observations=self._next_observations[:self._size, ...],
            dones=self._dones[:self._size, ...]
        )


def get_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
    )

def get_preprocessed_dataset(env, latent_action_dim):
    dataset = d4rl.qlearning_dataset(env)
    sample_size, _ = np.shape(dataset['actions'])
    latent_actions = np.random.uniform(low=-1, high=1, size=(sample_size, latent_action_dim))
    # latent_actions = np.zeros((sample_size, 1))
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        latent_actions=latent_actions, 
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
    )


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def parition_batch_train_test(batch, train_ratio):
    train_indices = np.random.rand(batch['observations'].shape[0]) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch


def subsample_batch(batch, size):
    indices = np.random.randint(batch['observations'].shape[0], size=size)
    return index_batch(batch, indices)


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated


def split_batch(batch, batch_size):
    batches = []
    length = batch['observations'].shape[0]
    keys = batch.keys()
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batches.append({key: batch[key][start:end, ...] for key in keys})
    return batches


def split_data_by_traj(data, max_traj_length):
    dones = data['dones'].astype(bool)
    start = 0
    splits = []
    for i, done in enumerate(dones):
        if i - start + 1 >= max_traj_length or done:
            splits.append(index_batch(data, slice(start, i + 1)))
            start = i + 1

    if start < len(dones):
        splits.append(index_batch(data, slice(start, None)))

    return splits


# def get_sarsa_dataset(data, max_traj_length=1000):
#     trajs = split_data_by_traj(data, max_traj_length)
#     observations = []
#     actions = []
#     next_observations = []
#     next_actions = []
#     rewards = []

#     for traj in trajs:
#         observations.append(traj['observations'][:-1])
#         actions.append(traj['actions'][:-1])
#         next_observations.append(traj['next_observations'][:-1])
#         next_actions.append(traj['actions'][1:])
#         rewards.append(traj['rewards'][:-1])
    
#     return dict(
#         observations=np.concatenate(observations),
#         actions=np.concatenate(actions),
#         next_observations=np.concatenate(next_observations),
#         rewards=np.concatenate(rewards),
#         next_actions=np.concatenate(next_actions),
#     )

def get_sarsa_dataset(env):
    dataset = env.get_dataset()
    dataset_cp = dict()
    for key in ['next_observations', 'observations', 'rewards', 'terminals', 'timeouts', 'actions']:
        dataset_cp[key] = dataset[key]
    dataset = dataset_cp

    dataset = d4rl.sequence_dataset(env, dataset)
    observations = []
    actions = []
    next_observations = []
    next_actions = []
    rewards = []
    dones = []
    for traj in dataset:
        observations.append(traj['observations'][:-1])
        actions.append(traj['actions'][:-1])
        next_observations.append(traj['next_observations'][:-1])
        next_actions.append(traj['actions'][1:])
        rewards.append(traj['rewards'][:-1])
        dones.append(traj['terminals'][:-1].astype(np.float32))
    
    return dict(
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        next_observations=np.concatenate(next_observations),
        rewards=np.concatenate(rewards),
        next_actions=np.concatenate(next_actions),
        dones=np.concatenate(dones),
    )

    


def get_top_dataset(data, filter_success=True, percentile=70.0, max_traj_length=1000):
    trajs = split_data_by_traj(data, max_traj_length)
    if filter_success:
        max_episode_return = np.max([traj['rewards'][-1] for traj in trajs])
        top_trajs = [traj for traj in trajs if traj['rewards'][-1] == max_episode_return]
    else:
        def compute_return(traj):
            episode_return = traj['rewards'][-1]
            return episode_return
        
        trajs.sort(key=compute_return)
        N = int(len(trajs) * percentile / 100)
        N = max(1, N)
        top_trajs = trajs[-N:]
    
    top_dataset = defaultdict(list)
    for traj in top_trajs:
        for key, val in traj.items():
            top_dataset[key].append(val)

    return dict(
        observations=np.concatenate(top_dataset['observations']),
        actions=np.concatenate(top_dataset['actions']),
        next_observations=np.concatenate(top_dataset['next_observations']),
        rewards=np.concatenate(top_dataset['rewards']),
        dones=np.concatenate(top_dataset['dones']).astype(np.float32),
    )

