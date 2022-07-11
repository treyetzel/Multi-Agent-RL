import torch
from torch import nn
import random


class QNet_FC(nn.Module):
    """
    Use this model for non-image based inputs
    """

    def __init__(self, obs, action_space):
        super(QNet_FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

    def forward(self, x):
        q_vals = self.fc(x)

        return q_vals

# TODO: Add model saving
    # def save(self):
    #     model_folder_path = './saved_models'
    #     if not os.path.exists(model_folder_path):
    #         os.makedirs(model_folder_path)
    #     # TODO: add a more dynamic file naming to reflect experiment
    #     file_name = "idqn_pong_agent_"

    #     for agent_i in range(self.num_agents):
    #         save_path = os.path.join(model_folder_path, file_name+str(agent_i))
    #         agent_model = getattr(self, 'agent_{}'.format(agent_i))
    #         torch.save(agent_model.state_dict(), save_path+".pth")
