from functools import total_ordering
import torch
from source.idqn import IDQN
from source.util.arguments import parser
from source.util.envs import get_env
args = parser.parse_args()
env, agent_names, is_image = get_env(args.env)

def test_idqn(agents, device):
    seeds = 0
    total_reward = 0
    agents_scores = {}
    for _ in range(10):
        env.reset(seeds)
        seeds += 1
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            if agent in agents_scores:
                agents_scores[agent] += reward
            else:
                agents_scores[agent] = reward
            if done:
                action = None
            else:
                agents.q_nets[agent].eval()
                q_vals = agents.q_nets[agent](
                    torch.Tensor(observation).unsqueeze(0).to(device)
                )
                action = q_vals.argmax().item()
            env.step(action)
            total_reward += reward

    avg_reward = total_reward / 10
    for agent in agents_scores:
        agents_scores[agent] /= 10

    return avg_reward, agents_scores

def test_hyperbolic_idqn(agents, device):
    seeds = 0
    total_reward = 0
    agents_scores = {}
    for _ in range(10):
        env.reset(seeds)
        seeds += 1
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            if agent in agents_scores:
                agents_scores[agent] += reward
            else:
                agents_scores[agent] = reward
            if done:
                action = None
            else:
                agents.q_nets[agent].eval()
                q_vals = agents.get_acting_q_vals(agent, torch.Tensor(observation).unsqueeze(0).to(device))
                action = q_vals.argmax().item()
            env.step(action)
            total_reward += reward

    avg_reward = total_reward / 10
    for agent in agents_scores:
        agents_scores[agent] /= 10

    return avg_reward, agents_scores


def display_model_idqn(path):
    env.reset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_names = env.agents
    agents = IDQN(
        env.observation_space(env.agent_selection),
        env.action_space(env.agent_selection).n,
        agent_names,
        device,
        is_image=is_image
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
                q_vals = agents.q_nets[agent](
                    torch.Tensor(observation).unsqueeze(0).to(device)
                )
                action = q_vals.argmax().item()
            env.step(action)
            env.render("human")
            total_reward += reward
    return total_reward / 10


if __name__ == "__main__":
    path = "./agents/idqn/models/idqn_model/"
    print(display_model_idqn(path))