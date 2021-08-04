import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.affine1 = nn.Linear(self.input_size, 128)
        self.affine2 = nn.Linear(128, self.output_size)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)