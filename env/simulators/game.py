class Game(object):
    def __init__(self, n_player, is_obs_continuous, is_act_continuous, game_name, agent_nums, obs_type):
        self.n_player = n_player
        self.current_state = None
        self.is_obs_continuous = is_obs_continuous
        self.is_act_continuous = is_act_continuous
        self.game_name = game_name
        self.agent_nums = agent_nums
        self.obs_type = obs_type

    def get_config(self, player_id):
        raise NotImplementedError

    def get_render_data(self, current_state):
        return current_state

    def set_current_state(self, current_state):
        raise NotImplementedError

    def is_terminal(self):
        raise NotImplementedError

    def get_next_state(self, joint_action):
        raise NotImplementedError

    def get_reward(self, joint_action):
        raise NotImplementedError

    def step(self, joint_action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def set_action_space(self):
        raise NotImplementedError