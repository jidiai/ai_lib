import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
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


class PG(object):
    def __init__(self, args):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.lr = args.c_lr
        self.gamma = args.gamma

        self.policy = Policy(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.saved_log_probs = []
        self.rewards = []

    def choose_action(self, observation, train=True):
        if train:
            state = torch.from_numpy(observation).float().unsqueeze(0)
            probs = self.policy(state)
            m = Categorical(probs)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
        else:
            state = torch.from_numpy(observation).float().unsqueeze(0)
            probs = self.policy(state)
            action = torch.argmax(probs)
        return action.item()

    # def get_buffer_data(self):


    def learn(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r[0] + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def save(self):
        torch.save(self.policy.state_dict(), 'policy_net.pth')

    def load(self, file):
        self.policy.load_state_dict(torch.load(file))


# policy = Policy()
# optimizer = optim.Adam(policy.parameters(), lr=1e-2)

"""
def main(args):
    base_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(base_dir))
    global env
    from EnvWrapper.classic_CartPole_v0 import Cartpole_v0
    env = Cartpole_v0()

    action_space = env.get_actionspace()
    observation_space = env.get_observationspace()

    global agent
    agent = pg(observation_space, action_space, args)

    for i_epoch in range(10000):
        state = env.reset()
        Gt = 0
        train_end = False
        for t in range(10000):  # Don't infinite loop while learning
            action = agent.choose_action(np.array(state))

            state, reward, done, _, _ = env.step(action)

            agent.rewards.append(reward)

            Gt += reward

            if done:
                print('i_epoch: ', i_epoch, 'Gt: ', '%.2f' % Gt)
                agent.learn()
                if i_epoch % args.evaluate_rate == 0 and i_epoch > 1:
                    Gt_real = evaluate(i_epoch)
                    if Gt_real > 199:
                        train_end = True
                break

        if train_end:
            print('1 save')
            agent.save()
            break

def evaluate(i_epoch):
    record = []
    for _ in range(100):
        state = env.reset()
        Gt_real = 0
        for t in count():
            action = agent.choose_action(state, train=False)
            next_state, reward, done, _, _ = env.step(action, train=False)
            state = next_state
            Gt_real += reward
            if done:
                record.append(Gt_real)
                break
    print('===============', 'i_epoch: ', i_epoch, 'Gt_real: ', '%.2f' % np.mean(record), record)
    return np.mean(record)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default = 0.001)
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (default: 0.99)')
    # evaluation
    parser.add_argument('--evaluate_rate', default=50)

    args = parser.parse_args()
    main(args)
    
"""