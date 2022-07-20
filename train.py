from test import test

import torch
import wandb
import os
import numpy as np
from source.idqn import IDQN
from source.util.arguments import parser
from source.util.envs import get_env, parallel_env

USE_WANDB = False

torch.set_default_dtype(torch.float32)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
configs = {}
for arg in vars(args):
    configs[arg] = getattr(args, arg)

if USE_WANDB:
    wandb.init(group=args.name, project="Multi-Agent-RL", entity="kevduong", config=configs)
    wandb.save("./source/models/idqn_models.py", base_path='./source/models/', policy="now")
    wandb.save("./source/idqn.py", base_path='./source/', policy="now")


env, agent_names, is_image = get_env(args.env)
env = parallel_env(env, args.num_envs)

max_paralell_steps = args.max_steps // args.num_envs
running_log_steps = 0  # number of steps since last log
warm_up_steps = args.warm_up_steps // args.num_envs
seed = args.seed
observations = env.reset(seed=seed)
torch.manual_seed(seed)
agents = IDQN(
    observation_space=env.observation_space,
    action_space=env.action_space.n,
    agent_names=agent_names,
    training_steps=int(max_paralell_steps * args.explore_rate),
    device=device,
    buffer_size=args.buffer_size,
    lr=args.lr,
    gamma=args.gamma,
    batch_size=args.batch_size,
    num_updates=args.num_updates,
    seed=seed,
)


# Training loop
for steps in range(1, max_paralell_steps + 1):
    with torch.no_grad():
        actions = agents.act(observations)

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
            # inverting dones, since we want Qprime to only be calculated if not done
            done_mask = 1 if dones_i[env_j] == 0 else 0
            # if obs is all 0, then agent has died (black_death supersuit wrapper)
            if done_mask == 1 and np.all((obs_prime_i[env_j] == 0)):
                done_mask = 0

            transition = (
                obs_i[env_j],
                actions_i[env_j],
                rews_i[env_j],
                obs_prime_i[env_j],
                done_mask,
            )
            agents.store(agent_names[agent_i], transition)

    observations = observations_prime

    if steps * args.num_envs >= warm_up_steps and steps % args.k_step == 0:
        agents.update()

    # increment number of steps per parallel environment
    running_log_steps += args.num_envs
    if running_log_steps >= args.log_steps or steps == max_paralell_steps:
        avg_rew, agent_scores = test(agents, device)
        
        if USE_WANDB:
            wandb.log({"test_avg_reward": avg_rew}, step=steps * args.num_envs)
            for agent in agent_names:
                wandb.log({f"{agent} average reward": agent_scores[agent]}, step=steps * args.num_envs)
        else:
            print(f"average rewards at step {steps*args.num_envs}: {avg_rew}")
            epsilon = agents.log()
            print(f"epsilon: {epsilon}")
            for agent in agent_names:
                print(f"{agent} average reward: {agent_scores[agent]}")
        running_log_steps = 0

path = "./saved_models/{}/".format(args.env)

env.close()
agents.save_model(path, seed)


