# -*- coding:utf-8  -*-
# 作者：zruizhi   
# 创建时间： 2020/7/10 10:24 上午   
# 描述：


class Game(object):
    def __init__(self, n_player):
        self.n_player = n_player
        self.current_state = None
    
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

