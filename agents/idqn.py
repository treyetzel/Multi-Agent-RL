import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from .models.idqn_models import QNet_FC
from .storage import ReplayBuffer

random.seed(0)
np.random.seed(0)


class IDQN:
    def __init__(
        self,
        observation_space,
        action_space,
        agent_names,
        device,
        buffer_size,
        batch_size,
        num_updates,
        lr=0.0005,
        gamma=0.99,
    ):
        self.agent_names = agent_names
        self.gamma = gamma
        self.action_space = action_space
        self.lr = lr
        self.q_nets = {}
        self.target_nets = {}
        self.buffers = {}
        self.optimizers = {}
        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.0005  # exponential decay rate for exploration prob
        self.batch_size = batch_size
        self.num_updates = num_updates
        # initialize agents
        for agent in agent_names:
            new_buffer = ReplayBuffer(buffer_size, device)
            q_net = QNet_FC(observation_space, self.action_space).to(device)
            target_net = QNet_FC(observation_space, self.action_space).to(device)
            target_net.load_state_dict(q_net.state_dict())
            optimizer = optim.Adam(q_net.parameters(), lr=lr)

            self.q_nets[agent] = q_net
            self.target_nets[agent] = target_net
            self.buffers[agent] = new_buffer
            self.optimizers[agent] = optimizer

    def _soft_update_target(self, local_model, target_model, tau):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def store(self, agent_name, transition):
        self.buffers[agent_name].put(transition)

    def act(self, observations, decay_step):
        explore_probability = self.epsilon_min + (
            self.epsilon - self.epsilon_min
        ) * np.exp(-self.epsilon_decay * decay_step)
        max_prob = random.random()
        actions = None
        if max_prob > explore_probability:
            for agent_i in range(len(self.agent_names)):
                obs_i = observations[
                    agent_i : observations.shape[0] : len(self.agent_names)
                ]
                q_vals_i = self.q_nets[self.agent_names[agent_i]].forward(
                    torch.Tensor(obs_i).to("cuda:0")
                )
                q_tuple = q_vals_i.max(dim=1)
                q_vals_i = q_tuple[0]
                action_i = q_tuple[1]
                action_i = action_i.view(q_vals_i.shape[0], 1)
                if actions is None:
                    actions = action_i
                else:
                    actions = torch.cat((actions, action_i), dim=1)
            actions = actions.detach().cpu().numpy()
        else:
            num_envs = int(observations.shape[0] / len(self.agent_names))
            actions = np.random.randint(
                self.action_space, size=(num_envs, len(self.agent_names))
            )

        return actions

    # TODO: Add these to IDQN params
    def update(self):
        for agent in self.agent_names:
            for update in range(self.num_updates):
                obs, a, r, obs_prime, done_mask = self.buffers[agent].sample(
                    self.batch_size
                )
                q_vals = self.q_nets[agent].forward(obs)
                q_a = q_vals.gather(1, a.unsqueeze(-1).long()).squeeze(-1)

                with torch.no_grad():
                    max_q_prime = (
                        self.target_nets[agent].forward(obs_prime).max(dim=1)[0]
                    )

                target = r + self.gamma * max_q_prime * done_mask
                loss = F.smooth_l1_loss(q_a, target)
                self.optimizers[agent].zero_grad()
                loss.backward()
                self.optimizers[agent].step()

                self._soft_update_target(
                    self.q_nets[agent], self.target_nets[agent], 0.01
                )

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for agent in self.agent_names:
            torch.save(self.q_nets[agent].state_dict(), path + agent + ".pt")

    def load_model(self, path):
        for agent in self.agent_names:
            self.q_nets[agent].load_state_dict(torch.load(path + agent + ".pt"))
            self.target_nets[agent].load_state_dict(torch.load(path + agent + ".pt"))
            # self.q_nets[agent].load_state_dict(torch.load(path + "agent_0_no_e_policy" + ".pt"))
            # self.target_nets[agent].load_state_dict(torch.load(path + "agent_0_no_e_policy" + ".pt"))
