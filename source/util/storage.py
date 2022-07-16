import torch
import random
import numpy as np
# SumTree
# https://adventuresinmachinelearning.com/sumtree-introduction-python/
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:

    e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    beta = 0.4  # importance-sampling, from initial value increasing to 1

    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, obs_shape, device):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):

        obs_ar = np.empty((n, *self.obs_shape.shape))
        a_ar = np.empty((n, 1))
        r_ar = np.empty((n, 1))
        obs_prime_ar = np.empty((n, *self.obs_shape.shape))
        done_ar = np.empty((n, 1))
        idxs = []
        priorities = []
        segment = self.tree.total() / n

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            obs, a, r, obs_prime, done = data
            obs_ar[i] = obs
            a_ar[i] = a
            r_ar[i] = r
            obs_prime_ar[i] = obs_prime
            done_ar[i] = done
            priorities.append(p)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return (
            torch.tensor(obs_ar, dtype=torch.float).to(self.device),
            torch.tensor(a_ar, dtype=torch.float).to(self.device),
            torch.tensor(r_ar, dtype=torch.float).to(self.device),
            torch.tensor(obs_prime_ar, dtype=torch.float).to(self.device),
            torch.tensor(done_ar, dtype=torch.float).to(self.device),
            idxs,
            is_weight,
        )

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
