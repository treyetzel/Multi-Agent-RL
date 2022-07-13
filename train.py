from test import test
import wandb
import numpy as np
import torch
from pettingzoo.butterfly import knights_archers_zombies_v10, cooperative_pong_v5
from pettingzoo.mpe import simple_v2
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit import (
    black_death_v3,
    concat_vec_envs_v1,
    flatten_v0,
    pettingzoo_env_to_vec_env_v1,
    color_reduction_v0,
)
from agents.idqn import IDQN
from util.arguments import parser

USE_WANDB = False


torch.set_default_dtype(torch.float32)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
configs = {}
for arg in vars(args):
    configs[arg] = getattr(args, arg)

if USE_WANDB:
    wandb.init(project="Multi-Agent-RL", entity="kevduong", config=configs)

# TODO: handle way to create env from args, need to handle for test as well

#  env = black_death_v3(knights_archers_zombies_v10.env(use_typemasks=True))
#  env = simple_v2.env(max_cycles=50, continuous_actions=False)
env = color_reduction_v0(cooperative_pong_v5.env(), mode="full")
env.reset()
agent_names = env.agents
# env = flatten_v0(env)
env = aec_to_parallel(env)
env = pettingzoo_env_to_vec_env_v1(env)
env = concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="gym")

# TODO: Finish seeding, all seeds are set to 0 right now, add as arg and go through all randoms (including numpy randoms)
torch.manual_seed(0)

# TODO: add .train() for nets https://discuss.pytorch.org/t/model-train-and-model-eval-vs-model-and-model-eval/5744

max_paralell_steps = args.max_steps // args.num_envs
running_log_steps = 0  # number of steps since last log
warm_up_steps = args.warm_up_steps // args.num_envs

observations = env.reset(seed=0)
agents = IDQN(
    observation_space=env.observation_space,
    action_space=env.action_space.n,
    agent_names=agent_names,
    training_steps=max_paralell_steps,
    device=device,
    buffer_size=args.buffer_size,
    lr=args.lr,
    gamma=args.gamma,
    batch_size=args.batch_size,
    num_updates=args.num_updates,
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

    if steps * args.num_envs >= warm_up_steps and steps % args.k_step == 0:
        agents.update()

    # increment number of steps per parallel environment
    running_log_steps += args.num_envs
    if running_log_steps >= args.log_steps or steps == max_paralell_steps:
        avg_rew = test(agents, device)
        print(f"average rewards at step {steps*args.num_envs}: {avg_rew}")
        if USE_WANDB:
            wandb.log({"test_avg_reward": avg_rew}, step=steps * args.num_envs)
        running_log_steps = 0
        epsilon = agents.log()
        print(f"epsilon: {epsilon}")

path = "./agents/idqn/models/idqn_model/"

env.close()
agents.save_model(path)
