from functools import total_ordering
import torch
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_v2
from supersuit import flatten_v0
from agents.idqn import IDQN

env = flatten_v0(knights_archers_zombies_v10.env(use_typemasks=True))

# env = flatten_v0(simple_v2.env(max_cycles=50, continuous_actions=False))


def test(agents, device):
    seeds = 0
    total_reward = 0
    for _ in range(10):
        env.reset(seeds)
        seeds += 1
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            if done:
                action = None
            else:
                q_vals = agents.q_nets[agent](torch.Tensor(observation).to(device))
                action = q_vals.argmax().item()
            env.step(action)
            total_reward += reward

    return total_reward / 10


def display_model(path):
    env.reset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_names = env.agents
    agents = IDQN(
        env.observation_space(env.agent_selection).shape[0],
        env.action_space(env.agent_selection).n,
        agent_names,
        device,
        buffer_size=None,
        batch_size=None,
        num_updates=None,
    )
    agents.load_model(path)
    total_reward = 0
    for _ in range(20):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            if done:
                action = None
            else:
                q_vals = agents.q_nets[agent](torch.Tensor(observation).to(device))
                action = q_vals.argmax().item()
            env.step(action)
            env.render("human")
            total_reward += reward

    return total_reward / 10


if __name__ == "__main__":
    path = "./agents/idqn/models/idqn_model/"
    print(display_model(path))
