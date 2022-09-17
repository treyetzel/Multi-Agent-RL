import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import math

from .models.idqn_models import QNet_FC_Hyperbolic, QNet_Nature_CNN
from .util.storage import PrioritizedReplayBuffer
from .HD.utils import compute_eval_gamma_intervals
from .HD.utils import integrate_q_values



class Hyperbolic_IDQN:
    def __init__(
        self,
        observation_space,
        action_space,
        agent_names,
        device,
        training_steps=None,
        buffer_size=1000,
        batch_size=32,
        num_updates=10,
        lr=0.0005,
        number_of_gammas=3,
        gamma_max=0.99,
        hyperbolic_exponent = 0.99,
        integral_estimate = 'lower',
        acting_policy = 'largest_gamma',
        is_image=False,
        seed=None,
    ):
        self.agent_names = agent_names
        self.device = device
        self.number_of_gammas = number_of_gammas
        self.gamma_max = gamma_max
        self.hyperbolic_exponent = hyperbolic_exponent
        self.integral_estimate = integral_estimate
        self.acting_policy = acting_policy
        self.action_space = action_space
        self.training_steps = training_steps  # used for epsilon annealing
        self.lr = lr
        self.q_nets = {}
        self.target_nets = {}
        self.buffers = {}
        self.optimizers = {}
        self.epsilon = 1.0  # exploration probability at start
        self.explore_probability = 1.0
        self.epsilon_min = 0.05  # minimum exploration probability
        self.batch_size = batch_size
        self.num_updates = num_updates
        self.is_image = is_image
        
        if seed != None:
            random.seed(seed)
            np.random.seed(seed)
        
        # hyperbolic stuff
        self.eval_gammas = compute_eval_gamma_intervals(gamma_max, hyperbolic_exponent, number_of_gammas)
        self.gammas = [math.pow(gamma, hyperbolic_exponent) for gamma in self.eval_gammas]

        # initialize agents
        for agent in agent_names:
            new_buffer = PrioritizedReplayBuffer(buffer_size, observation_space, device)
            if is_image:
                raise NotImplementedError("Have not implemented hyperbolic CNN")
            else:
                q_net = QNet_FC_Hyperbolic(observation_space.shape[0], self.action_space, self.number_of_gammas).to(device)
                target_net = QNet_FC_Hyperbolic(observation_space.shape[0], self.action_space, self.number_of_gammas).to(device)
                
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
        obs, a, r, obs_prime, done_mask = transition
        if self.is_image:
            raise NotImplementedError("Have not implemented hyperbolic CNN")
        else:
            obs = torch.Tensor(obs).to(self.device)
            obs_prime = torch.Tensor(obs_prime).to(self.device)

        with torch.no_grad():
            q_vals = self.get_acting_q_vals(agent_name, obs)
            target_max_normal_q_vals = self.target_nets[agent_name].forward(obs_prime).max(dim=1).values


        errors = []

        q_val_a = q_vals[a].cpu().numpy()

        for gamma_num in range(self.number_of_gammas):
            target = r + self.gammas[gamma_num] * target_max_normal_q_vals[gamma_num].cpu().numpy() * done_mask
            error = abs(target - q_val_a)
            errors.append(error)

        avg_error = sum(errors)/len(errors)

        self.buffers[agent_name].add(avg_error, transition)

    def act(self, observations):
        if self.explore_probability > self.epsilon_min:
            self.explore_probability -= (
                self.epsilon - self.epsilon_min
            ) / self.training_steps

        max_prob = random.random()
        actions = None
        if max_prob > self.explore_probability:
            for agent_i in range(len(self.agent_names)):
                obs_i = observations[
                    agent_i : observations.shape[0] : len(self.agent_names)
                ]
                if obs_i.ndim == 3:
                    obs_i = torch.Tensor(obs_i).unsqueeze(1).to(self.device)
                else:
                    obs_i = torch.Tensor(obs_i).to(self.device)
                
                q_vals_i = self.get_acting_q_vals(self.agent_names[agent_i], obs_i)
                
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

    def update(self):
        for agent in self.agent_names:
            for update in range(self.num_updates):
                obs, a, r, obs_prime, done_mask, idxs, is_weights = self.buffers[
                    agent
                ].sample(self.batch_size)

                if obs.dim() == 3:
                    obs = obs.unsqueeze(1)
                    obs_prime = obs_prime.unsqueeze(1)

                self.q_nets[agent].train()


                # one_hot_actions of replay actions
                one_hot_actions = F.one_hot(a.squeeze(-1).to(torch.int64), num_classes=len(self.get_acting_q_vals(agent, obs)[0]))
                
                chosen_q_values = []
                for i in range(self.number_of_gammas):
                    chosen_q_value = torch.sum(self.q_nets[agent].forward(obs)[i] * one_hot_actions, dim=-1)
                    chosen_q_values.append(chosen_q_value)
                
                
                with torch.no_grad():
                    # NEED TO VERIFY THIS IS HOW TO CALCULATE Q-VALS FOR HYPER-IDQN

                    # (double dqn) get actions from online network to use on target values 
                    next_obs_acting_q_vals = self.get_acting_q_vals(agent, obs_prime).argmax(dim=1).unsqueeze(1)
                    # [256, 1]
                    
                    target_value_of_next_obs = self.target_nets[agent].forward(obs_prime)
                    # [3, 256, 6]
                    
                    target_q_primes_selected = target_value_of_next_obs.gather(2, next_obs_acting_q_vals.repeat(self.number_of_gammas,1,1).long())
                    # [3, 256, 1]
                    
                    

                targets = []
                for gamma_num in range(self.number_of_gammas):
                    target = (r + self.gammas[gamma_num] * target_q_primes_selected[gamma_num] * done_mask).squeeze(-1)
                    targets.append(target)
                # targets in form [tensor([256]), tensor(256)... for the # of gammas]
                
                # calculate errors for replay buffer
                errors = torch.abs(targets[0] - chosen_q_values[0]).detach().cpu().numpy()
                for i in range(1, self.number_of_gammas):
                    errors += torch.abs(targets[i] - chosen_q_values[i]).detach().cpu().numpy()
                errors = errors / self.number_of_gammas
                
                
                # update priorities/replay buffer
                for i in range(self.batch_size):
                    idx = idxs[i]
                    self.buffers[agent].update(idx, errors[i]) 
                
                # calculate loss
                loss = 0
                for gamma_num in range(self.number_of_gammas):
                    gamma_loss = (
                        torch.Tensor(is_weights).to(self.device)
                        * F.smooth_l1_loss(chosen_q_values[gamma_num], targets[gamma_num], reduction="none")
                    ).mean()
                    loss += gamma_loss
                loss = loss / self.number_of_gammas

                self.optimizers[agent].zero_grad()
                loss.backward()
                self.optimizers[agent].step()
                self._soft_update_target(
                    self.q_nets[agent],
                    self.target_nets[agent],
                    0.05,  # TODO: Add this as a hyperparameter maybe? (tau) for updating target nets
                )

    def log(self):
        return self.explore_probability

    def save_model(self, path, seed):
        if not os.path.exists(path):
            os.makedirs(path)
        for agent in self.agent_names:
            torch.save(self.q_nets[agent].state_dict(), path + agent + "_s-" + str(seed) +".pt")

    def load_model(self, path):
        for agent in self.agent_names:
            self.q_nets[agent].load_state_dict(torch.load(path + agent + ".pt"))
            self.target_nets[agent].load_state_dict(torch.load(path + agent + ".pt"))

    def get_acting_q_vals(self, agent_name, obs):
        if self.acting_policy == 'largest_gamma':
            return self.q_nets[agent_name].forward(obs)[-1]
        elif self.acting_policy== 'hyperbolic':
            raise NotImplementedError("Having issues with hyperbolic action selection")
            original_output = self.q_nets[agent_name].forward(obs)
            net_out = self.q_nets[agent_name].forward(obs)
            if obs.ndim == 1:
                net_out = net_out.unsqueeze(0)
            else:
                net_out = net_out.transpose(0, 1)


            hyperbolic_q_values_list = np.zeros(len(self.q_nets[agent_name].forward(obs)[0]))


            for action_num in range(len(self.q_nets[agent_name].forward(obs)[0])):
                q_values = self.q_nets[agent_name].forward(obs)[:,action_num]

                if len(q_values.size()) != 1:
                    print(self.q_nets[agent_name].forward(obs))
                    print(self.q_nets[agent_name].forward(obs).size())
                    print(obs.ndim)

                    x = self.q_nets[agent_name].forward(obs).transpose(0, 1)
                    print(x)
                    print(x.size())
                    print(x.ndim)
                
                hyperbolic_q_values_list[action_num] = integrate_q_values(q_values, self.integral_estimate, self.eval_gammas, self.number_of_gammas, self.gammas)
            return torch.from_numpy(hyperbolic_q_values_list)
        else:
            raise NotImplementedError()
    