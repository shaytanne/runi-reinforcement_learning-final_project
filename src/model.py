import numpy as np
import torch
import torch.nn as nn

class MiniGridCNN(nn.Module):
    """
    CNN wrapper for the network(s) used by the agents
    Input: (<batch_size>, num_channels=1, height=84, width=84)
    Output: Q values for each action
    """

    def __init__(self, input_shape: np.ndarray, num_actions: int):
        super().__init__()

        # parse input dimensions
        num_channels, height, width = input_shape
        
        # CONV layers:
        self.visual_features = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # find out dimensions of feature map after CONV
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, height, width)
            conv_output = self.visual_features(dummy_input)
            self.features_dims = conv_output.view(1, -1).size(1)
        
        # FC layers:
        self.value_predictor = nn.Sequential(
           nn.Flatten(),
           nn.Linear(self.features_dims, 512),
           nn.ReLU(),
           nn.Linear(512, num_actions)
        )
        
    def forward(self, x) -> torch.Tensor:
        x = self.visual_features(x)
        x = self.value_predictor(x)
        return x

