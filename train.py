import random
import time
from test import test

import numpy as np
import torch
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_v2
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit import (
    black_death_v3,
    concat_vec_envs_v1,
    flatten_v0,
    pettingzoo_env_to_vec_env_v1,
)

from agents.idqn import IDQN
from util.arguments import parser

args = parser.parse_args()

torch.set_default_dtype(torch.float32)

# env = black_death_v3(knights_archers_zombies_v10.env(use_typemasks=True))
env = simple_v2.env(max_cycles=50, continuous_actions=False)
env.reset()
# getting list of agents to store batches for corresponding agents
agent_names = env.agents
env = aec_to_parallel(flatten_v0(env))
env = pettingzoo_env_to_vec_env_v1(env)
env = concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="gym")
torch.manual_seed(0)

# TODO: add .train() for nets https://discuss.pytorch.org/t/model-train-and-model-eval-vs-model-and-model-eval/5744

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

observations = env.reset()
agents = IDQN(
    env.observation_space.shape[0],
    env.action_space.n,
    agent_names,
    device,
    buffer_size=args.buffer_size,
)


max_paralell_steps = args.max_steps // args.num_envs
running_log_steps = 0  # number of steps since last log
warm_up_steps = args.warm_up_steps // args.num_envs

# Training loop
for steps in range(1, max_paralell_steps + 1):
    with torch.no_grad():
        actions = agents.act(
            observations, decay_step=0 if steps < warm_up_steps else steps
        )

    observations_prime, rewards, dones, info = env.step(actions.flatten())

    # splitting transitions for separate agents, and storing in replay buffer
    for agent_i in range(len(agent_names)):
        obs_i = observations[agent_i : observations.shape[0] : len(agent_names)]
        actions_i = actions[:, agent_i]
        rews_i = rewards[agent_i : rewards.shape[0] : len(agent_names)]
        obs_prime_i = observations_prime[
            agent_i : observations.shape[0] : len(agent_names)
        ]
        dones_i = dones[agent_i : dones.shape[0] : len(agent_names)]

        for env_j in range(args.num_envs):
            # q_val = r + gamma * (max_a' Q(s', a') * done mask)
            done_mask = 1 if dones_i[env_j] == 0 else 0
            transition = (
                obs_i[env_j],
                actions_i[env_j],
                rews_i[env_j],
                obs_prime_i[env_j],
                done_mask,
            )
            agents.store(agent_names[agent_i], transition)

    observations = observations_prime

    if steps * args.num_envs >= warm_up_steps:
        agents.update()

    # increment number of steps per parallel environment
    running_log_steps += args.num_envs
    if running_log_steps >= args.log_steps or steps == max_paralell_steps:
        print(f"average rewards at step {steps*args.num_envs}: {test(agents, device)}")
        running_log_steps = 0

path = "./agents/idqn/models/idqn_model/"

env.close()
agents.save_model(path)
