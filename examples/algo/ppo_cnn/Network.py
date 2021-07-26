import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T


class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=None, is_matrix=False):
        super(Actor, self).__init__()
        self.is_matrix = is_matrix
        if not self.is_matrix:
            assert (hidden_size!=None),"Please set the value of hidden_size"
            self.fc1 = nn.Linear(state_space, hidden_size)
            self.action_head = nn.Linear(hidden_size, action_space)
        else:
            self.a_conv = self.conv_net(state_space,action_space,hidden_size)

    def conv_net(self, state_space, action_space,hiddden_size):
        h,w,c = state_space
        model = nn.Sequential(
            nn.Conv2d(c,10,kernel_size=7, padding=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(10,20,kernel_size=5, padding=2, stride=2,groups=10),
            nn.ReLU(),
            nn.Conv2d(20,20,kernel_size=3, padding=1, groups=10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2800,256),
            nn.ReLU(),
            nn.Linear(256,action_space)
        )
        return model

    def forward(self, x):
        if not self.is_matrix:
            x = F.relu(self.fc1(x))
            action_prob = F.softmax(self.action_head(x), dim=1)
        else:
            x = self.a_conv(x)
            action_prob = F.softmax(x, dim=1)        
        return action_prob     


class Critic(nn.Module):
    def __init__(self, state_space, output_size, hidden_size=None, is_matrix=False):
        super(Critic, self).__init__()
        self.is_matrix = is_matrix
        if not is_matrix:
            assert (hidden_size!=None),"Please set the value of hidden_size"
            self.fc1 = nn.Linear(state_space, hidden_size)
            self.state_value = nn.Linear(hidden_size, output_size)
        else:
            self.c_conv = self.conv_net(state_space, output_size, hidden_size)

    def conv_net(self, state_space, output_size,hiddden_size):
        h,w,c = state_space
        model = nn.Sequential(
            nn.Conv2d(c,10,kernel_size=7, padding=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(10,20,kernel_size=5, padding=2, stride=2,groups=10),
            nn.ReLU(),
            nn.Conv2d(20,20,kernel_size=3, padding=1, groups=10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2800,256),
            nn.ReLU(),
            nn.Linear(256,output_size)
        )
        return model

    def forward(self, x):
        if not self.is_matrix:
            x = F.relu(self.fc1(x))
            value = self.state_value(x)
        else:
            value = self.c_conv(x)
        return value