# -*- coding:utf-8  -*-
# Time  : 2021/10/25 下午2:28
# Author: Yahui Cui


import copy
from gym.utils import seeding
from env.simulators.game import Game
from env.obs_interfaces.observation import *
from utils.discrete import Discrete


class ChessAndCard(Game, DictObservation):
    def __init__(self, conf, seed=None):
        super(ChessAndCard, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                           conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.seed = None
        self.done = False
        self.dones = {}
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])

        env_name = conf["game_name"]
        import_path = "from pettingzoo.classic import " + env_name + " as env_imported"
        exec(import_path)
        func_name = "env_imported"
        self.env_core = None
        self.env_core = eval(func_name).env()

        if self.env_core is None:
            raise Exception("ChessAndCard env_core is None!")

        self.episode_count = 30 if self.game_name in ['texas_holdem_no_limit_v3', 'texas_holdem_v3'] else 1
        self.won = {}
        self.n_return = [0] * self.n_player
        self.step_cnt = 0
        self.step_cnt_episode = 0
        self.done = False
        self.seed = seed
        self.env_core.seed(self.seed)
        self.env_core.reset()
        self.player_id_map, self.player_id_reverse_map = self.get_player_id_map(self.env_core.agents)

        # set up action spaces
        self.new_action_spaces = self.load_action_space()
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space
        self.input_dimension = self.env_core.observation_spaces

        # set up first all_observes
        obs, _, _, _ = self.env_core.last()
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        self.init_info = self.get_info_after()

    def reset(self):
        self.step_cnt = 0
        self.step_cnt_episode = 0
        self.done = False
        self.env_core.reset()
        obs, _, _, _ = self.env_core.last()
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        self.init_info = self.get_info_after()
        self.won = {}
        self.n_return = [0] * self.n_player
        return self.all_observes

    def reset_per_episode(self):
        self.step_cnt_episode = 0
        self.done = False
        self.env_core.reset()
        obs, _, _, _ = self.env_core.last()
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        self.init_info = self.get_info_after()
        return self.all_observes

    def step(self, joint_action):
        self.step_cnt_episode += 1
        self.is_valid_action(joint_action)
        info_before = self.step_before_info()
        joint_action_decode = self.decode(joint_action)
        self.env_core.step(joint_action_decode)
        obs, reward, episode_done, info_after = self.env_core.last()
        info_after = self.get_info_after()
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        # print("debug all observes ", type(self.all_observes[0]["obs"]))
        self.set_n_return()
        self.step_cnt += 1
        if episode_done:
            self.episode_count -= 1
            if self.episode_count > 0:
                self.all_observes = self.reset_per_episode()
                info_after = self.init_info
        done = self.is_terminal()
        return self.all_observes, reward, done, info_before, info_after

    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:
            raise Exception("Input joint action dimension should be {}, not {}.".format(
                self.n_player, len(joint_action)))

        current_player_id = self.player_id_map[self.env_core.agent_selection]
        if (self.env_core.agent_selection in self.env_core.agents) and \
                (not self.env_core.dones[self.env_core.agent_selection]):
            if joint_action[current_player_id] is None or joint_action[current_player_id][0] is None:
                raise Exception("Action of current player is needed. Current player is {}, {}".format(
                    current_player_id, self.env_core.agent_selection))

        for i in range(self.n_player):
            if joint_action[i] is None or joint_action[i][0] is None:
                continue
            if len(joint_action[i][0]) != self.joint_action_space[i][0].n:
                raise Exception("The input action dimension for player {} should be {}, not {}.".format(
                    i, self.joint_action_space[i][0].n, len(joint_action[i][0])))

    def step_before_info(self, info=''):
        return info

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        # if not self.env_core.agents:
        #     self.done = True
        #
        # if all(self.env_core.dones.values()):
        #     self.done = True

        if self.episode_count == 0:
            self.done = True

        return self.done

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def load_action_space(self):
        origin_action_spaces = self.env_core.action_spaces
        new_action_spaces = {}
        for key, action_space in origin_action_spaces.items():
            changed_key = self.player_id_map[key]
            new_action_spaces[changed_key] = Discrete(action_space.n)

        return new_action_spaces

    def set_action_space(self):
        action_space = [[self.new_action_spaces[i]] for i in range(self.n_player)]
        return action_space

    def check_win(self):
        if self.all_equals(self.n_return):
            return '-1'

        index = []
        max_n = max(self.n_return)
        for i in range(len(self.n_return)):
            if self.n_return[i] == max_n:
                index.append(i)

        if len(index) == 1:
            return str(index[0])
        else:
            return str(index)

    def decode(self, joint_action):
        if self.env_core.agent_selection not in self.env_core.agents or \
                self.env_core.dones[self.env_core.agent_selection]:
            return None
        current_player_id = self.player_id_map[self.env_core.agent_selection]
        if joint_action[current_player_id] is None or joint_action[current_player_id][0] is None:
            return None
        joint_action_decode = joint_action[current_player_id][0].index(1)
        return joint_action_decode

    def set_n_return(self):
        for player_key, player_reward in self.env_core.rewards.items():
            player_id = self.player_id_map[player_key]
            self.n_return[player_id] += player_reward

    def get_player_id_map(self, player_keys):
        player_id_map = {}
        player_id_reverse_map = {}
        for i, key in enumerate(player_keys):
            player_id_map[key] = i
            player_id_reverse_map[i] = key
        return player_id_map, player_id_reverse_map

    def create_seed(self):
        seed = seeding.create_seed(None, max_bytes=4)
        return seed

    def set_seed(self, seed=None):
        self.env_core.seed(seed)
        self.seed = seed

    def get_all_observes(self):
        all_observes = []
        is_new_episode = 1 if self.step_cnt_episode == 0 else 0
        for i in range(self.n_player):
            player_name = self.player_id_reverse_map[i]
            each_obs = copy.deepcopy(self.current_state)
            if self.game_name in ['texas_holdem_no_limit_v3', 'texas_holdem_v3', 'leduc_holdem_v3']:
                if self.player_id_map[self.env_core.agent_selection] == i:
                    each = {"obs": each_obs, "is_new_episode": is_new_episode,
                            "current_move_player": self.env_core.agent_selection,
                            "controlled_player_index": i, "controlled_player_name": player_name}
                else:
                    each = {"obs": None, "is_new_episode": is_new_episode,
                            "current_move_player": self.env_core.agent_selection,
                            "controlled_player_index": i, "controlled_player_name": player_name}
            else:
                each = {"obs": each_obs, "is_new_episode": is_new_episode,
                        "current_move_player": self.env_core.agent_selection,
                        "controlled_player_index": i, "controlled_player_name": player_name}
            all_observes.append(each)

        return all_observes

    def all_equals(self, list_to_compare):
        return len(set(list_to_compare)) == 1

    def get_info_after(self):
        info_after = ''
        if self.game_name in ['texas_holdem_no_limit_v3', 'texas_holdem_v3']:
            info_after = {}
            for i in range(self.n_player):
                temp_info = copy.deepcopy(self.env_core.env.env.env.env.game.get_state(i))
                if self.game_name in ['texas_holdem_no_limit_v3']:
                    for action_index, action in enumerate(temp_info['legal_actions']):
                        temp_info['legal_actions'][action_index] = str(action)
                    temp_info['stage'] = str(temp_info['stage'])
                info_after[self.player_id_reverse_map[i]] = temp_info

        if self.game_name in ['leduc_holdem_v3']:
            info_after = {}
            for i in range(self.n_player):
                temp_info = self.env_core.env.env.env.env.env.game.get_state(i)
                info_after[self.player_id_reverse_map[i]] = copy.deepcopy(temp_info)

        return info_after
