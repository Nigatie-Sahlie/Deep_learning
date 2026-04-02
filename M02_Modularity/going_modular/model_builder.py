#  it is the time to create or build a model. the model will be simple for educational purposes like tinyVGG model.

import torch
from torch import nn

# create class of tinyVGG model

class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:    # it is the constructor of the class and it defines the architecture of the model.
        super().__init__()                                                                 # Connect my model with pyTorch system. it is important for tracking the model and its parameters.

        # Define the model architecture(the layers of the model).
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,                     
                      padding=0),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*63 * 63,
                      out_features=output_shape)
        )

        # Define the data flow through the model (the forward method).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.Classifier(x)
        return x