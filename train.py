from pettingzoo.butterfly import knights_archers_zombies_v10
import numpy as np
from pettingzoo.utils.conversions import aec_to_parallel
import random
from agents.idqn_2 import IDQN
import time
from supersuit import flatten_v0, pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1, black_death_v3
import torch

torch.set_default_dtype(torch.float32)


env = black_death_v3(knights_archers_zombies_v10.env(use_typemasks=True))
env.reset()
# getting list of agents to store batches for corresponding agents
env_agents = env.agents

env = aec_to_parallel(flatten_v0(env))
env = pettingzoo_env_to_vec_env_v1(env)
env = concat_vec_envs_v1(env, 4, num_cpus=0, base_class='gym')
torch.manual_seed(0)


episodes = 2
total_reward = 0
done = False

observations = env.reset()
print(env_agents)
print(observations.shape)
# agents = IDQN(env)


print(type(observations))
act = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


observations = env.reset()
for steps in range(10):

    observations, rewards, dones, info = env.step(act)
    print(observations.shape)

    dictt = {}
    for i in range(len(env_agents)):
        dictt[env_agents[i]] = observations[i:observations.shape[0]:len(env_agents)]

    for agent in env_agents:
        print("=========================")
        print(agent)
        print(dictt[agent].shape)
        print("=========================")

    # while completed_episodes < episodes:
    #     env.reset()
    #     for current_agent in env.agent_iter():
    #         obs, _, done, _ = env.last()

    #         if done:
    #             action = None
    #             env.step(action)
    #         else:
    #             with torch.no_grad():
    #                 action = agents.act(
    #                     current_agent, torch.Tensor(obs), max_action=False)

    #             env.step(action)

    #             reward = env.rewards[current_agent]
    #             obs_prime = env.observe(current_agent)
    #             done_mask = env.observe(current_agent)

    #             print(done_mask)
    #             transition = (obs, action, reward, obs_prime, done_mask)
    #             agents.store(current_agent, transition)

    #         if completed_episodes >= 1:
    #             agents.update(current_agent)

    #     completed_episodes += 1
    # env.close()

    # print("Average total reward", total_reward / episodes)
