# -*- coding:utf-8  -*-
# Time  : 2021/10/25 下午2:28
# Author: Yahui Cui


import copy
import itertools

from gym.utils import seeding
from env.simulators.game import Game
from env.obs_interfaces.observation import *
from utils.discrete import Discrete

import pygame
import os
import numpy as np
import math
import random
import warnings

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

        if conf.get('env_cfg', None) is None:
            self.env_core = eval(func_name).env()
        else:
            self.env_core = eval(func_name).env(**conf['env_cfg'])

        # for Texas Hold'em details, see https://github.com/datamllab/rlcard/blob/master/docs/games.md#no-limit-texas-holdem

        if self.env_core is None:
            raise Exception("ChessAndCard env_core is None!")

        self.episode_count = 30 if self.game_name in ['texas_holdem_no_limit_v3', 'texas_holdem_v3',
                                                      "texas_holdem_v4",
                                                      "texas_holdem_no_limit_v5"] else 1
        if self.game_name in ['mahjong_v4']:
            self.episode_count = 1 #math.factorial(self.n_player)

        self.won = {}
        self.n_return = [0] * self.n_player
        self.step_cnt = 0
        self.step_cnt_episode = 0
        self.game_cnt = 1
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

        other_cfg = conf.get('other_cfg', {})
        self.switch_seats = other_cfg.get('switch_seats', False)
        if self.switch_seats:
            self.seats_permutation = list(itertools.permutations([i for i in range(self.n_player)], 4))
            self.seat_permutation_idx = 0
            self.player_seats = dict(zip(self.env_core.agents, self.seats_permutation[0]))      # player_name: seat_idx
            self.same_hand = other_cfg['same_hand']         #same hand for each sub-game
            if self.same_hand and self.seed is None:        #if no seed has been specified, we generate one
                self.seed = random.randint(0,10000000)
                self.env_core.seed(self.seed)
                self.env_core.reset()

        self.payoff = dict(zip(self.env_core.agents, [0]*self.n_player))    #np.zeros(self.n_player)
        self.action_masks_dict = dict(zip(self.env_core.agents, [np.ones(i[0].n) for i in self.action_dim]))

        # set up first all_observes
        obs, _, _, _ = self.env_core.last()
        self.current_state = obs
        self.action_masks_dict[self.env_core.agent_selection] = obs['action_mask']
        self.all_observes = self.get_all_observes()
        self.init_info = self.get_info_after()
        self.screen = None


    def reset(self):            #not used currently
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
        self.game_cnt += 1
        self.done = False
        self.action_masks_dict = dict(zip(self.env_core.agents, [np.ones(i[0].n) for i in self.action_dim]))

        if self.switch_seats:       #switch clockwise
            self.seat_permutation_idx += 1

            self.player_seats = dict(zip(list(self.player_seats.keys()),
                                         self.seats_permutation[self.seat_permutation_idx]))

            if self.same_hand:
                self.env_core.seed(self.seed)

        self.env_core.reset()
        obs, _, _, _ = self.env_core.last()
        self.current_state = obs
        self.action_masks_dict[self.env_core.agent_selection] = obs['action_mask']

        self.all_observes = self.get_all_observes()
        self.init_info = self.get_info_after()
        return self.all_observes

    def step(self, joint_action):
        self.step_cnt_episode += 1
        joint_action=self.is_valid_action(joint_action)
        info_before = self.step_before_info()
        joint_action_decode = self.decode(joint_action)
        self.env_core.step(joint_action_decode)
        obs, reward, episode_done, info_after = self.env_core.last()
        info_after = self.get_info_after()
        self.current_state = obs
        self.action_masks_dict[self.env_core.agent_selection] = obs['action_mask']

        self.all_observes = self.get_all_observes()
        # print("debug all observes ", type(self.all_observes[0]["obs"]))
        self.step_cnt += 1
        if episode_done:
            self.episode_count -= 1
            if self.game_name in ['mahjong_v4']:
                seats_payoff = np.array(self.env_core.env.env.env.env.env.get_payoffs())
                seats_player = dict(zip(self.player_seats.values(), self.player_seats.keys()))
                for seat in range(len(seats_payoff)):
                    player_id = seats_player[seat]
                    self.payoff[player_id] += seats_payoff[seat]

            if self.episode_count > 0:
                self.all_observes = self.reset_per_episode()
                info_after = self.init_info
        done = self.is_terminal()
        self.set_n_return()
        if done:
            print(f'Final payoff = {self.payoff}')
        return self.all_observes, reward, done, info_before, info_after

    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:
            raise Exception("Input joint action dimension should be {}, not {}.".format(
                self.n_player, len(joint_action)))

        current_player_id = self.player_id_map[self.env_core.agent_selection]
        if not self.switch_seats:
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
        else:
            reversed_player_seats = dict(zip(self.player_seats.values(), self.player_seats.keys()))
            current_player = reversed_player_seats[current_player_id]
            current_idx = int(current_player[-1])
            if len(joint_action[current_idx][0])!=self.joint_action_space[current_player_id][0].n:
                raise Exception("The input action dimension for player {} should be {}, not {}.".format(
                    current_player_id, self.joint_action_space[current_player_id][0].n,
                    len(joint_action[current_idx][0])))

            if (np.array(joint_action[current_idx][0]) * self.action_masks_dict[self.env_core.agent_selection]).sum() == 0:
                warnings.warn(f"The action of player {current_player} has illegal action, "
                              f"input action = {joint_action[current_idx][0]} but the legal action should be {self.action_masks_dict[self.env_core.agent_selection]}")
                action_mask = self.action_masks_dict[self.env_core.agent_selection]
                rand_action = np.random.multinomial(1, np.array(action_mask) / sum(action_mask))
                joint_action[current_idx] = [list(rand_action)]

        return joint_action


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
        current_seat_id = self.player_id_map[self.env_core.agent_selection]           #seat id

        if not self.switch_seats and \
                (joint_action[current_seat_id] is None or joint_action[current_seat_id][0] is None):
            return None

        if self.switch_seats:
            seats_player = dict(zip(self.player_seats.values(), self.player_seats.keys()))
            current_player = seats_player[current_seat_id]
            current_idx = int(current_player[-1])
            current_action = joint_action[current_idx][0].index(1)
            return current_action
        else:
            joint_action_decode = joint_action[current_seat_id][0].index(1)

        return joint_action_decode

    def set_n_return(self):
        if self.game_name in ['mahjong_v4']:
            self.n_return = list(self.payoff.values())
            if 1 in self.n_return:      #someone wins, make the payoff zero-sum
                self.n_return = [3*i if i==1 else i for i in self.n_return]
            return

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
        self.env_core.reset()

    def get_all_observes(self):
        all_observes = []
        is_new_episode = 1 if self.step_cnt_episode == 0 else 0
        for i in range(self.n_player):
            player_name = self.player_id_reverse_map[i]
            each_obs = copy.deepcopy(self.current_state)
            if self.game_name in ['texas_holdem_no_limit_v3', 'texas_holdem_v3',
                                  'leduc_holdem_v3', 'texas_holdem_no_limit_v5',
                                  "texas_holdem_v4", 'mahjong_v4']:
                if self.player_id_map[self.env_core.agent_selection] == i:
                    each = {"obs": each_obs, "is_new_episode": is_new_episode,      #
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

        if self.switch_seats:
            switched_all_observes = []
            for pid in self.env_core.agents:
                current_seat = self.player_seats[pid]
                _obs = all_observes[current_seat]
                switched_all_observes.append(_obs)

            return switched_all_observes

        return all_observes

    def all_equals(self, list_to_compare):
        return len(set(list_to_compare)) == 1

    def get_info_after(self):
        info_after = ''
        if self.game_name in ['texas_holdem_no_limit_v3', 'texas_holdem_v3',
                              "texas_holdem_v4","texas_holdem_no_limit_v5"]:
            info_after = {}
            for i in range(self.n_player):
                temp_info = copy.deepcopy(self.env_core.env.env.env.env.game.get_state(i))
                if self.game_name in ['texas_holdem_no_limit_v3', 'texas_holdem_no_limit_v5']:
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


    def render(self):
        #only support mahjong now
        assert self.game_name in ['mahjong_v4']
        screen_height = 1000
        screen_width = 1000

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)

        tile_size = screen_height/15
        tile_short_edge = tile_size*(500/667)
        bg_color = (7, 99, 36)
        white = (255, 255, 255)
        cyan = (0,255,255)
        red = (255,0,0)
        self.screen.fill(bg_color)

        def get_image(path):
            from os import path as os_path
            cwd = os_path.dirname(__file__)
            image = pygame.image.load(cwd + '/' + path)
            return image

        def get_font(path, size):
            from os import path as os_path
            cwd = os_path.dirname(__file__)
            font = pygame.font.Font((cwd + '/' + path), size)
            return font

        def display_text(text, coord, font, color, loc='center'):
            _text = font.render(text, True,color)
            _textRect = _text.get_rect()
            setattr(_textRect, loc, coord)
            self.screen.blit(_text, _textRect)

        font = get_font(os.path.join(f'Texas_Holdem/font/Minecraft.ttf'), 18)
        display_text(f"Game {self.game_cnt}", (screen_width - 150, 10), font, white, loc='topleft')
        display_text(f"Payoff: {list(self.payoff.values())}", (screen_width-200, 30), font, white, loc='topleft')

        base_env=self.env_core.env.env.env.env
        for i,player in enumerate(self.env_core.possible_agents):
            state = base_env.env.game.get_state(base_env._name_to_int(player))
            # current_hand = state['current_hand']          #has bugs causing same hand for all players
            current_hand = base_env.env.game.players[i].hand
            current_hand = [i.get_str() for i in current_hand]
            sorted_hand = sorted(current_hand)
            for j,tile in enumerate(sorted_hand):
                tile_img = get_image(os.path.join(f"images/mahjong/{tile}.png"))
                tile_img = pygame.transform.scale(tile_img, (int(tile_short_edge),
                                                             int(tile_size)))
                if i%2==0:
                    self.screen.blit(tile_img,
                                     (screen_width/2-len(sorted_hand)/2*tile_short_edge+j*tile_short_edge,
                                      screen_height/8*(i*3+1)-tile_size/2))
                else:
                    if i==1:
                        tile_img = pygame.transform.rotate(tile_img, 90)
                        loc = (screen_width/8*7-tile_size/2, screen_height/2-int(tile_short_edge)/2-len(sorted_hand)/2*tile_short_edge+tile_short_edge*j)
                    else:
                        tile_img = pygame.transform.rotate(tile_img, -90)
                        loc = (screen_width/8-tile_size/2, screen_height/2-int(tile_short_edge)/2-len(sorted_hand)/2*tile_short_edge+tile_short_edge*j)
                    self.screen.blit(tile_img,
                                     loc)

            current_pile = state['players_pile'][i]
            # if len(current_pile)>0:
            pile_gap = 10
            for pile_idx, pile in enumerate(current_pile):
                for tile_idx, tile in enumerate(pile):
                    tile_img = get_image(os.path.join(f"images/mahjong/{tile.get_str()}.png"))
                    tile_img = pygame.transform.scale(tile_img, (int(tile_short_edge),
                                                                 int(tile_size)))
                    if i==0:
                        loc = (100+tile_short_edge*tile_idx+pile_idx*(4*tile_short_edge+pile_gap), 20)
                    elif i==1:
                        tile_img = pygame.transform.rotate(tile_img, 90)
                        loc = (screen_width-70, 200+tile_short_edge*tile_idx+pile_idx*(4*tile_short_edge+pile_gap))
                    elif i==2:
                        loc = (100+tile_short_edge*tile_idx+pile_idx*(4*tile_short_edge+pile_gap), screen_height-70)
                    elif i==3:
                        tile_img = pygame.transform.rotate(tile_img, -90)
                        loc = (20, 100+tile_short_edge*tile_idx+pile_idx*(4*tile_short_edge+pile_gap))

                    self.screen.blit(tile_img, loc)

            # player_seat = self.player_seats[player]
            seats_player = dict(zip(self.player_seats.values(), self.player_seats.keys()))

            # player_text = f"Seat {i}"
            # player_text = font.render(player_text, True, white)
            # player_textRect = player_text.get_rect()
            if i == 0:
                display_text(f"Seat {i} - {seats_player[i]}", (screen_width/2, 50), font, white, loc='center')
            elif i == 1:
                display_text(f"Seat {i}",(screen_width-80, screen_height/2),font, white, loc='topleft')
                display_text(f'{seats_player[i]}', (screen_width-80, screen_height/2+30), font, white, loc='topleft')
            elif i == 2:
                display_text(f'Seat {i} - {seats_player[i]}', (screen_width/2, screen_height-50), font, white, loc='center')
            elif i ==3:
                display_text(f"Seat {i}", (10, screen_height/2), font, white, loc='topleft')
                display_text(f'{seats_player[i]}', (10, screen_height/2+30), font, white, loc='topleft')

        remaining_deck = len(self.env_core.env.env.env.env.env.game.dealer.deck)
        display_text(f"{remaining_deck} left", (20,20), font, white, loc='topleft')


        table_tiles = [i.get_str() for i in state['table']]
        tiles_per_row = 10
        tiles_rows = len(table_tiles)//tiles_per_row

        init_loc = (230,200)

        for idx, tile in enumerate(table_tiles):
            row_idx = (idx//tiles_per_row)
            y = init_loc[1]+tile_size*row_idx
            col_idx = (idx%tiles_per_row)
            x = init_loc[0]+tile_short_edge*col_idx
            loc = (x, y)

            tile_img = get_image(os.path.join(f"images/mahjong/{tile}.png"))
            tile_img = pygame.transform.scale(tile_img, (int(tile_short_edge),
                                                         int(tile_size)))
            self.screen.blit(tile_img, loc)

        pygame.display.update()





