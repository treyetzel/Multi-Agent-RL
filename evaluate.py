import numpy as np
import torch
import torch.nn as nn
import gym
import random
def evaluate_coop():
    """
    With evaluating co-op environments, we can typically just test
    with one model/algorithm, and then compare with other test runs
    of different models/algorithms. For example, running a test run with all agents on
    hyperbolic, and then running another one with non-hyperbolic to see which one
    compares better. The reward return is simpler, as it should represent
    a final reward for the whole team.
    """
    raise NotImplementedError #TODO

def evaluate_competitive():
    """
    evaluating on competitve environments is going to be a little different,
    here we would like to use perhaps two models/algorithms in the same test run.
    In a competitve setting, such as pong, we may want one agent using hyperbolic
    and another agent using non-hyperbolic. Rewards are computed as lists depending
    on how many teams, in the scenario of pong, rewards are presented as: [agent1_r,agent2_r] 
    so comparing agents with two different algorithms, we can see which one scores more.
    """
    raise NotImplementedError #TODO

# STILL WORK IN PROGRESS, BUT LOADS MODEL NOW
# AGENT 1 IS THE LOADED MODEL, AGENT 2 IS RANDOM
class eval_agents_prototype:
    def __init__(self, observation_space, action_space):
        print(action_space[0].n)
        n_obs = observation_space[0].shape[0]
        self.agent1 = nn.Sequential(nn.Linear(n_obs, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, action_space[0].n))
        self.action_space = action_space[0].n
    
    def load(self,model_path):
        self.agent1.load_state_dict(torch.load(model_path))
        
    # just a temporary solution
    # The idea is sample from different agent models,
    # and then combine their actions and return
    def sample_action(self, obs, epsilon):
        q_value = [torch.empty(obs.shape[0], )]
        q_value[0] = self.agent1(obs[:, 0, :]).unsqueeze(1)
        out =torch.cat(q_value, dim=1)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        actions = action[0].data.cpu().numpy().tolist()
        # appending random agent actions
        actions.append(float(random.randint(0, self.action_space - 1)))
        #actions.append(0.0)
        #print(actions)
        return actions
        
# figure out a better way to load saved models
def eval_competitive_prototype(env, num_episodes, agents):
    score = np.zeros(env.n_agents)
    for episode_i in range(num_episodes):
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        while not all(done):
            action = agents.sample_action(torch.Tensor(state).unsqueeze(0), epsilon=0)
            next_state, reward, done, info = env.step(action)
            if(all(done)):
                score += np.array(reward)
                print(score)
            state = next_state
            
env = gym.make("ma_gym:PongDuel-v0")

agents = eval_agents_prototype(env.observation_space, env.action_space)
agents.load("./saved_models/idqn_pong_agent_0.pth")

eval_competitive_prototype(env, 100, agents)