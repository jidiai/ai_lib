import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        """
        x: [batch_size, ...]
        """
        return x.view(x.size(0), -1)


class CNN_encoder(nn.Module):
    """
    pre-defined CNN hyperparameters
    """
    def __init__(self, hidden_dim = 128):
        super(CNN_encoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0),  # [batch, 32, 13,18]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=0),  # [batch, 64,5,7]
            nn.BatchNorm2d(32),
            nn.ReLU(),  # [batch, 64, 5, 7]
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # [batch, 3, 5]
            Flatten(),
            nn.Linear(1120, hidden_dim),
            nn.ReLU()
        )

    def forward(self, state):
        """
        state: [batch, 60, 80, 3]
        """
        assert len(state.shape) == 4
        state = state/255.
        state = state.permute(0, 3, 1, 2)

        return self.net(state)


#another class of encoder is the LSTM net....
