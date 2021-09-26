import torch

class DQN(object):
    def __init__(self, args, Critic):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space
        self.hidden_size = args.hidden_size
        self.critic_eval = Critic(self.state_dim,  self.action_dim, self.hidden_size)

    def choose_action(self, observation):
        inference_output = self.inference(observation)
        return inference_output

    def inference(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
        action = torch.argmax(self.critic_eval(observation)).item()
        return {"action": action}

    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))