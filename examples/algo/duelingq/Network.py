import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        x2 = F.relu(self.linear1(x))

        # value
        y1 = self.linear2(x1)
        # advantage
        y2 = self.linear3(x2)
        # y2.mean(dim=1,keepdim=True)：dim=1按行取均值；keepdim=True时，输出与输入维度相同，仅仅时输出在求均值的维度上元素个数变为1。keepdim=False时，输出比输入少一个维度，就是指定的dim求均值的维度。
        x3 = y1 + y2 - y2.mean(dim=1, keepdim=True)

        x = x3
        return x