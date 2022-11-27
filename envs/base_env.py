class BaseEnv:
    def __init__(self, id, seed):
        self.id = id
        self.seed = seed
        self.agent_ids = None
        self.step_ctr = 0

    def reset(self):
        pass

    def step(self):
        pass

    def get_episode_stats(self):
        pass

    def is_terminated(self):
        pass
