import collections
import torch
import random


class ReplayBuffer:
    def __init__(self, buffer_size, device):
        self.device = device
        self.buffer = collections.deque(maxlen=buffer_size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done)

        return (
            torch.tensor(s_lst, dtype=torch.float).to(self.device),
            torch.tensor(a_lst, dtype=torch.float).to(self.device),
            torch.tensor(r_lst, dtype=torch.float).to(self.device),
            torch.tensor(s_prime_lst, dtype=torch.float).to(self.device),
            torch.tensor(done_mask_lst, dtype=torch.float).to(self.device),
        )

    def size(self):
        return len(self.buffer)
