import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from itertools import count
import argparse
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from network import Critic

class DQN(object):
    def __init__(self, obs_space, action_space, args):

        self.state_dim = obs_space
        self.action_dim = action_space

        self.hidden_size = args.hidden_size
        self.lr = args.c_lr
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        self.critic_eval = Critic(self.state_dim,  self.action_dim, self.hidden_size)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.buffer = []

        # epsilon
        self.eps = args.epsilon
        self.eps_end = args.epsilon_end # todo
        self.eps_end = 0
        self.eps_delay = 1 / (args.max_episode * 100) # todo

        self.learn_step_counter = 0
        self.target_replace_iter = args.target_replace

    def choose_action(self, observation, train=True):
        if train:
            self.eps = max(self.eps_end, self.eps - self.eps_delay)
            if random.random() < self.eps:
                action = random.randrange(self.action_dim)
            else:
                observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
                action = torch.argmax(self.critic_eval(observation)).item()
        else:
            observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
            action = torch.argmax(self.critic_eval(observation)).item()
        return action

    def store_transition(self, transition):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        obs, action, reward, obs_, done = zip(*samples)

        obs = torch.tensor(obs, dtype=torch.float).squeeze()
        action = torch.tensor(action, dtype=torch.long).view(self.batch_size, -1)
        reward = torch.tensor(reward, dtype=torch.float).view(self.batch_size, -1).squeeze()
        obs_ = torch.tensor(obs_, dtype=torch.float).squeeze()
        done = torch.tensor(done, dtype=torch.float).view(self.batch_size, -1).squeeze()


        q_eval = self.critic_eval(obs).gather(1, action)
        q_next = self.critic_target(obs_).detach()
        q_target = (reward + self.gamma * q_next.max(1)[0] * (1 - done)).view(self.batch_size, 1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.learn_step_counter += 1

        return loss

    def save(self):
        torch.save(self.critic_eval.state_dict(), 'critic_net.pth')

    def load(self, file):
        self.critic.load_state_dict(torch.load(file))

def main(args):
    base_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(base_dir))
    global env
    # from EnvWrapper.classic_MountainCar_v0 import MountainCar_v0
    # env = MountainCar_v0()
    from EnvWrapper.classic_CartPole_v0 import Cartpole_v0
    env = Cartpole_v0()

    action_space = env.get_actionspace()
    observation_space = env.get_observationspace()

    seed = 1
    torch.manual_seed(seed)
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

    global agent
    agent = DQN(observation_space, action_space, args)

    for i_epoch in range(10000):
        state = env.reset()
        Gt = 0
        train_end = False
        for t in count():
            action = agent.choose_action(state)

            next_state, reward, done, _, _= env.step(action)

            trans = Transition(state, action, reward, np.array(next_state), done)

            if len(agent.buffer) >= agent.batch_size:
                agent.learn()

            agent.store_transition(trans)
            state = next_state

            Gt += reward

            if done:
                print('i_epoch: ', i_epoch, 'Gt: ', '%.2f' % Gt,'epi: ', '%.2f' % agent.eps)
                if i_epoch % args.evaluate_rate == 0 and i_epoch > 1:
                    Gt_real = evaluate(i_epoch)
                    if Gt_real > 199.9:
                        train_end = True
                break

        if train_end:
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
    print('===============', 'i_epoch: ', i_epoch, 'Gt_real: ', '%.2f' % np.mean(record))
    return np.mean(record)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="classic_CartPole-v0", type=str)
    parser.add_argument('--max_episodes', default=500, type=int)
    parser.add_argument('--algo', default="ppo", type=str, help="dqn/ppo/a2c")

    parser.add_argument('--buffer_capacity', default=int(10240), type=int)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.005, type=float)
    parser.add_argument('--c_lr', default=0.005, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--clip_param', default=0.2, type=int)
    parser.add_argument('--max_grad_norm', default=0.5, type=int)
    parser.add_argument('--ppo_update_time', default=10, type=int)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--hidden_size', default=100)
    parser.add_argument('--max_episode', default=1000, type=int)
    parser.add_argument('--target_replace', default=100)

    # exploration
    parser.add_argument('--epsilon', default=0.2) # cartpole 0.2 # mountaincar 1
    parser.add_argument('--epsilon_end', default=0.05) # cartpole 0.05 # mountaincar 0.05

    # evaluation
    parser.add_argument('--evaluate_rate', default=50)

    # pre-train

    args = parser.parse_args()

    main(args)
    print("end")

