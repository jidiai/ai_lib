import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from networks.network_td3 import Actor, Critic
from common.buffer import Replay_buffer as buffer
from pathlib import Path
import sys
import os

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))


max_action = 2.0
min_Val = torch.tensor(1e-7).float() # min value


def get_trajectory_property():
    return ["action", "logits"]


class TD3():
    def __init__(self, args):

        self.state_dim = args.obs_space
        self.action_dim = 1

        self.actor = Actor(self.state_dim, self.action_dim, max_action)
        self.actor_target = Actor(self.state_dim, self.action_dim, max_action)
        self.critic_1 = Critic(self.state_dim, self.action_dim)
        self.critic_1_target = Critic(self.state_dim, self.action_dim)
        self.critic_2 = Critic(self.state_dim, self.action_dim)
        self.critic_2_target = Critic(self.state_dim, self.action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.update_freq = args.update_freq
        self.eps = 0.2

        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_delay = args.policy_delay
        self.exploration_noise = args.exploration_noise

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def choose_action(self, observation, train=True):
        inference_output = self.inference(observation, train)
        if train:
            self.add_experience(inference_output)
        return inference_output

    def inference(self, state, train=True):
        state = torch.tensor(state, dtype=torch.float).view(1, -1)
        action = self.actor(state).cpu().data.numpy().flatten()
        action += np.random.normal(0, self.exploration_noise, size=self.action_dim)
        action = action.clip(- max_action, max_action)
        logits = self.actor(state).detach().numpy()
        return {"action": action,
                "logits": logits}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def learn(self):

        data_length = len(self.memory.item_buffers["rewards"].data)
        if data_length < self.buffer_size:
            return

        for i in range(self.update_freq):
            data = self.memory.sample(self.batch_size)
            transitions = {
                "o_0": np.array(data['states']),
                "o_next_0": np.array(data['states_next']),
                "r_0": np.array(data['rewards']).reshape(-1, 1),
                "u_0": np.array(data['action']),
                "d_0": np.array(data['dones']).reshape(-1, 1),
            }

            state = torch.tensor(transitions["o_0"], dtype=torch.float)
            next_state = torch.tensor(transitions["o_next_0"], dtype=torch.float)
            action = torch.tensor(transitions["u_0"], dtype=torch.float).view(self.batch_size, -1)
            reward = torch.tensor(transitions["r_0"], dtype=torch.float).view(self.batch_size, -1)
            done = torch.tensor(transitions["d_0"], dtype=torch.float).view(self.batch_size, -1)
            # Select next action according to target policy:

            noise = torch.ones_like(action).data
            noise = torch.tensor(noise, dtype=torch.float)
            noise = noise.normal_(0., self.policy_noise)
            noise = noise.clamp(- self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates:
            if i % self.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_critic_1_path = os.path.join(base_path, "critic_1" + str(episode) + ".pth")
        torch.save(self.critic_1.state_dict(), model_critic_1_path)
        model_critic_1_target_path = os.path.join(base_path, "critic_1_target" + str(episode) + ".pth")
        torch.save(self.critic_1_target.state_dict(), model_critic_1_target_path)

        model_critic_2_path = os.path.join(base_path, "critic_2" + str(episode) + ".pth")
        torch.save(self.critic_2.state_dict(), model_critic_2_path)
        model_critic_2_target_path = os.path.join(base_path, "critic_2_target" + str(episode) + ".pth")
        torch.save(self.critic_2_target.state_dict(), model_critic_2_target_path)

        model_actor_path = os.path.join(base_path, "actor" + str(episode) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)
        model_actor_target_path = os.path.join(base_path, "actor_target" + str(episode) + ".pth")
        torch.save(self.actor_target.state_dict(), model_actor_target_path)

    def load(self, save_path):
        self.actor.load_state_dict(torch.load(str(save_path) + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(str(save_path) + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(str(save_path) + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(str(save_path) + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(str(save_path) + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(str(save_path) + 'critic_2_target.pth'))

