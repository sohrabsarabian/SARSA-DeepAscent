import torch.nn as nn


class QNetwork(nn.Module):
    """
    The Q-network model used in the Deep SARSA algorithm.
    """

    def __init__(self, state_dims, num_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, state):
        """
        Forward pass of the Q-network.
        """
        return self.layers(state)
