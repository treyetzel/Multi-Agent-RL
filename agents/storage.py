import collections
import torch
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, device):
        self.device = device
        self.buffer = collections.deque(maxlen=buffer_size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        obs_shape = mini_batch[0][0].shape[0]
        obs_ar = np.empty((n, obs_shape))
        a_ar = np.empty((n, 1))
        r_ar = np.empty((n, 1))
        obs_prime_ar = np.empty((n, obs_shape))
        done_ar = np.empty((n, 1))

        for i, transition in enumerate(mini_batch):
            s, a, r, s_prime, done = transition
            obs_ar[i] = s
            a_ar[i] = a
            r_ar[i] = r
            obs_prime_ar[i] = s_prime
            done_ar[i] = done

        return (
            torch.tensor(obs_ar, dtype=torch.float).to(self.device),
            torch.tensor(a_ar, dtype=torch.float).to(self.device),
            torch.tensor(r_ar, dtype=torch.float).to(self.device),
            torch.tensor(obs_prime_ar, dtype=torch.float).to(self.device),
            torch.tensor(done_ar, dtype=torch.float).to(self.device),
        )

    def size(self):
        return len(self.buffer)
