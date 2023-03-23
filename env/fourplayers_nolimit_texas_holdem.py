# -*- coding:utf-8  -*-

import copy
import warnings

from gym.utils import seeding
from env.simulators.game import Game
from env.obs_interfaces.observation import *
from utils.discrete import Discrete
from pettingzoo.utils.agent_selector import agent_selector
from rlcard.games.nolimitholdem import Action
import pygame
import os
import random

import numpy as np
import rlcard
from gym import spaces
import math
import itertools

try:
    from pettingzoo import AECEnv
    from pettingzoo.utils import wrappers
except:
    raise ImportError("The multiplayer Texas hold'em has dependencies on pettingzoo")


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

"""
action id:  0   Folde
            1   Check
            2   Call
            3   Raise Half Pot
            4   Raise Full Pot
            5   All in
"""

# RLCard version conflist (action space conflits !!!)


class RLCardBase(AECEnv):
    """
    Borrow from pettingzoo
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, name, num_players, obs_shape):
        super().__init__()
        self.name = name
        self.num_players = num_players
        config = {'allow_step_back': False,
                  'seed': None,
                  'game_num_players': num_players}

        self.env = rlcard.make(name, config)
        self.screen = None
        if not hasattr(self, "agents"):
            self.agents = [f'player_{i}' for i in range(num_players)]
        self.possible_agents = self.agents[:]

        dtype = self.env.reset()[0]['obs'].dtype
        if dtype == np.dtype(np.int64):
            self._dtype = np.dtype(np.int8)
        elif dtype == np.dtype(np.float64):
            self._dtype = np.dtype(np.float32)
        else:
            self._dtype = dtype

        self.observation_spaces = self._convert_to_dict(
            [spaces.Dict({'observation': spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=self._dtype),
                          'action_mask': spaces.Box(low=0, high=1, shape=(self.env.num_actions,),
                                                    dtype=np.int8)}) for _ in range(self.num_agents)])
        self.action_spaces = self._convert_to_dict([spaces.Discrete(self.env.num_actions) for _ in range(self.num_agents)])

        self.action_record = {}
        self.last_agent_selection = None


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        config = {'allow_step_back': False,
                  'seed': seed,
                  'game_num_players': self.num_players}
        self.env = rlcard.make(self.name, config)

    def _scale_rewards(self, reward):
        return reward

    def _int_to_name(self, ind):
        return self.possible_agents[ind]

    def _name_to_int(self, name):
        return self.possible_agents.index(name)

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def observe(self, agent):
        obs = self.env.get_state(self._name_to_int(agent))
        observation = obs['obs'].astype(self._dtype)

        legal_moves = self.next_legal_moves
        action_mask = np.zeros(self.env.num_actions, 'int8')
        for i in legal_moves:
            action_mask[i] = 1

        return {'observation': observation, 'action_mask': action_mask}

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        self.action_record[self.agent_selection] = action
        self.last_agent_selection = self.agent_selection

        obs, next_player_id = self.env.step(action)
        next_player = self._int_to_name(next_player_id)
        self._last_obs = self.observe(self.agent_selection)
        if self.env.is_over():
            self.rewards = self._convert_to_dict(self._scale_rewards(self.env.get_payoffs()))
            self.next_legal_moves = []
            self.dones = self._convert_to_dict([True if self.env.is_over() else False for _ in range(self.num_agents)])
        else:
            self.next_legal_moves = obs['legal_actions']
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_player
        self._accumulate_rewards()
        self._dones_step_first()

    def reset(self):
        obs, player_id = self.env.reset()

        self.init_dealer_id = self.env.game.dealer_id
        self.small_blind_player = (self.init_dealer_id+1)%self.env.num_players
        self.big_blind_player = (self.init_dealer_id+2)%self.env.num_players

        self.agents = self.possible_agents[:]
        self.agent_selection = self._int_to_name(player_id)
        self.rewards = self._convert_to_dict([0 for _ in range(self.num_agents)])
        self._cumulative_rewards = self._convert_to_dict([0 for _ in range(self.num_agents)])
        self.dones = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = self._convert_to_dict([{'legal_moves': []} for _ in range(self.num_agents)])
        self.next_legal_moves = list(sorted(obs['legal_actions']))
        self._last_obs = obs['obs']

    def render(self, mode='human'):
        raise NotImplementedError()

    def close(self):
        pass



class RenderEnv(RLCardBase):

    metadata = {'render.modes': ['human', 'rgb_array'], "name": "texas_holdem_no_limit_v5"}

    def __init__(self, num_players=2):
        super().__init__("no-limit-holdem", num_players, (54,))
        self.observation_spaces = self._convert_to_dict([spaces.Dict(
            {'observation': spaces.Box(low=np.zeros(54, ), high=np.append(np.ones(52, ), [100, 100]), dtype=np.float32),
             'action_mask': spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8)}) for _ in range(self.num_agents)])

    def render(self, mode='human'):

        def calculate_width(self, screen_width, i):
            return int((screen_width / (np.ceil(len(self.possible_agents) / 2) + 1) * np.ceil((i + 1) / 2)) + (tile_size * 31 / 616))

        def calculate_offset(hand, j, tile_size):
            return int((len(hand) * (tile_size * 23 / 56)) - ((j) * (tile_size * 23 / 28)))

        def calculate_height(screen_height, divisor, multiplier, tile_size, offset):
            return int(multiplier * screen_height / divisor + tile_size * offset)

        def display_text(text, coord, font, color, loc='center'):
            _text = font.render(text, True,color)
            _textRect = _text.get_rect()
            setattr(_textRect, loc, coord)
            self.screen.blit(_text, _textRect)


        screen_height = 1000
        screen_width = 1500  #int(screen_height * (1 / 20) + np.ceil(len(self.possible_agents) / 2) * (screen_height * 1 / 2))

        if self.screen is None:
            if mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))
        if mode == "human":
            pygame.event.get()

        # Setup dimensions for card size and setup for colors
        tile_size = screen_height * 2 / 10

        bg_color = (7, 99, 36)
        white = (255, 255, 255)
        cyan = (0,255,255)
        red = (255,0,0)
        self.screen.fill(bg_color)

        chips = {0: {'value': 10000, 'img': 'ChipOrange.png', 'number': 0},
                 1: {'value': 5000, 'img': 'ChipPink.png', 'number': 0},
                 2: {'value': 1000, 'img': 'ChipYellow.png', 'number': 0},
                 3: {'value': 100, 'img': 'ChipBlack.png', 'number': 0},
                 4: {'value': 50, 'img': 'ChipBlue.png', 'number': 0},
                 5: {'value': 25, 'img': 'ChipGreen.png', 'number': 0},
                 6: {'value': 10, 'img': 'ChipLightBlue.png', 'number': 0},
                 7: {'value': 5, 'img': 'ChipRed.png', 'number': 0},
                 8: {'value': 1, 'img': 'ChipWhite.png', 'number': 0}}

        stake_text_loc = [(150,100), (150, 800),(1300, 100), (1300, 800)]
        font = get_font(os.path.join('Texas_Holdem/font', 'Minecraft.ttf'), 24)

        display_text(f"Game {self.game_cnt}", (screen_width-350, screen_height/2), font, white, loc='topleft')
        display_text(f"Payoff: {list(self.payoff.values())}", (screen_width-350, screen_height/2+40), font, white, loc='topleft')

        display_text(f"Current Pot {self.env.game.dealer.pot}", (200, screen_height/2), font, white, loc='topleft')

        # Load and blit all images for each card in each player's hand
        for i, player in enumerate(self.possible_agents):
            state = self.env.game.get_state(self._name_to_int(player))
            for j, card in enumerate(state['hand']):
                # Load specified card
                card_img = get_image(os.path.join('Texas_Holdem/img', card + '.png'))
                card_img = pygame.transform.scale(card_img, (int(tile_size * (142 / 197)), int(tile_size)))
                # Players with even id go above public cards
                if i % 2 == 0:
                    self.screen.blit(card_img, ((calculate_width(self, screen_width, i) - calculate_offset(state['hand'], j, tile_size)), calculate_height(screen_height, 4, 1, tile_size, -1)))
                # Players with odd id go below public cards
                else:
                    self.screen.blit(card_img, ((calculate_width(self, screen_width, i) - calculate_offset(state['hand'], j, tile_size)), calculate_height(screen_height, 4, 3, tile_size, 0)))

            # Load and blit text for player name

            # player_seat = self.player_seats[player]
            seat_player = dict(zip(self.player_seats.values(), self.player_seats.keys()))
            player_id = seat_player[i]

            if i==self.small_blind_player:
                _text = f'Seat {str(i)} - small - ({player_id})'
                _color=white
            elif i==self.big_blind_player:
                _text = f'Seat {str(i)} - big - ({player_id})'
                _color=red
            else:
                _text = f"Seat {str(i)} - ({player_id})"
                _color=white

            text = font.render(_text, True, _color)
            textRect = text.get_rect()
            if i % 2 == 0:
                textRect.center = ((screen_width / (np.ceil(len(self.possible_agents) / 2) + 1) * np.ceil((i + 1) / 2)), calculate_height(screen_height, 4, 1, tile_size, -(22 / 20)))
            else:
                textRect.center = ((screen_width / (np.ceil(len(self.possible_agents) / 2) + 1) * np.ceil((i + 1) / 2)), calculate_height(screen_height, 4, 3, tile_size, (23 / 20)))
            self.screen.blit(text, textRect)

            # Load and blit number of poker chips for each player
            font = get_font(os.path.join('Texas_Holdem/font', 'Minecraft.ttf'), 24)
            text = font.render(str(state['my_chips']), True, white)
            textRect = text.get_rect()

            # Calculate number of each chip
            total = (state['my_chips'])
            height = 0
            for key in chips:
                num = total / chips[key]['value']
                chips[key]['number'] = int(num)
                total %= chips[key]['value']

                chip_img = get_image(os.path.join('Texas_Holdem/img', chips[key]['img']))
                chip_img = pygame.transform.scale(chip_img, (int(tile_size / 2), int(tile_size * 16 / 45)))

                # Blit poker chip img
                for j in range(0, int(chips[key]['number'])):
                    if i % 2 == 0:
                        self.screen.blit(chip_img, ((calculate_width(self, screen_width, i) + tile_size * (8 / 10)), calculate_height(screen_height, 4, 1, tile_size, -1 / 2) - ((j + height) * tile_size / 15)))
                    else:
                        self.screen.blit(chip_img, ((calculate_width(self, screen_width, i) + tile_size * (8 / 10)), calculate_height(screen_height, 4, 3, tile_size, 1 / 2) - ((j + height) * tile_size / 15)))
                height += chips[key]['number']

            # Blit text number
            if i % 2 == 0:
                textRect.center = ((calculate_width(self, screen_width, i) + tile_size * (21 / 20)), calculate_height(screen_height, 4, 1, tile_size, -1 / 2) - ((height + 1) * tile_size / 15))
            else:
                textRect.center = ((calculate_width(self, screen_width, i) + tile_size * (21 / 20)), calculate_height(screen_height, 4, 3, tile_size, 1 / 2) - ((height + 1) * tile_size / 15))
            self.screen.blit(text, textRect)

            display_text(f"Stakes = {state['stakes'][i]}", stake_text_loc[i], font, white, loc='topleft')

            if player in self.action_record:
                action_name = Action(self.action_record[player]).name
                if player == self.last_agent_selection:
                    _color=cyan
                else:
                    _color=white

                display_text(f'{action_name}', (stake_text_loc[i][0], stake_text_loc[i][1]+30), font, white, loc='topleft')


        # Load and blit public cards
        for i, card in enumerate(state['public_cards']):
            card_img = get_image(os.path.join('Texas_Holdem/img', card + '.png'))
            card_img = pygame.transform.scale(card_img, (int(tile_size * (142 / 197)), int(tile_size)))
            if len(state['public_cards']) <= 3:
                self.screen.blit(card_img, (((((screen_width / 2) + (tile_size * 31 / 616)) - calculate_offset(state['public_cards'], i, tile_size)), calculate_height(screen_height, 2, 1, tile_size, -(1 / 2)))))
            else:
                if i <= 2:
                    self.screen.blit(card_img, (((((screen_width / 2) + (tile_size * 31 / 616)) - calculate_offset(state['public_cards'][:3], i, tile_size)), calculate_height(screen_height, 2, 1, tile_size, -21 / 20))))
                else:
                    self.screen.blit(card_img, (((((screen_width / 2) + (tile_size * 31 / 616)) - calculate_offset(state['public_cards'][3:], i - 3, tile_size)), calculate_height(screen_height, 2, 1, tile_size, 1 / 20))))

        if mode == "human":
            pygame.display.update()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return np.transpose(observation, axes=(1, 0, 2)) if mode == "rgb_array" else None


def WrappedEnv(**kwargs):
    env = RenderEnv(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class FourPlayersNoLimitTexasHoldem(Game, DictObservation):
    def __init__(self, conf, seed=None):
        super(FourPlayersNoLimitTexasHoldem, self).__init__(conf['n_player'], conf['is_obs_continuous'],
                                                            conf['is_act_continuous'],conf['game_name'],
                                                            conf['agent_nums'], conf['obs_type'])
        self.dones = {}
        self.max_step = int(conf["max_step"])

        env_name = conf["game_name"]

        self.env_core = WrappedEnv(**conf['env_cfg'])

        # for Texas Hold'em details, see https://github.com/datamllab/rlcard/blob/master/docs/games.md#no-limit-texas-holdem

        self.episode_count = 1 #math.factorial(self.n_player)

        self.won = {}
        self.n_return = [0] * self.n_player
        self.step_cnt = 0
        self.step_cnt_episode = 0
        self.game_cnt = 1
        self.env_core.env.env.env.game_cnt = self.game_cnt
        self.done = False
        self.seed = seed
        self.env_core.seed(self.seed)
        self.env_core.reset()
        self.player_id_map, self.player_id_reverse_map = self.get_player_id_map(self.env_core.agents)

        other_cfg = conf.get('other_cfg', {})
        self.switch_seats = other_cfg.get('switch_seats', False)
        if self.switch_seats:
            self.seats_permutation = list(itertools.permutations([i for i in range(self.n_player)], 4))
            self.seats_permutation_idx = 0
            self.player_seats = dict(zip(self.env_core.agents, self.seats_permutation[0]))
            self.env_core.env.env.env.player_seats = self.player_seats
            self.same_hand = other_cfg['same_hand']
            if self.same_hand and self.seed is None:
                self.seed =  random.randint(0,10000000)
                self.env_core.seed(self.seed)
                self.env_core.reset()

        self.payoff = dict(zip(self.env_core.agents, [0]*self.n_player))#np.zeros(self.n_player)
        self.env_core.env.env.env.payoff = self.payoff

        # set up action spaces
        self.new_action_spaces = self.load_action_space()
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space
        self.input_dimension = self.env_core.observation_spaces

        self.action_masks_dict = dict(zip(self.env_core.agents, [np.ones(i[0].n) for i in self.action_dim]))

        # set up first all_observes
        obs, _, _, _ = self.env_core.last()
        self.action_masks_dict[self.env_core.agent_selection] = obs['action_mask']
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
        self.game_cnt += 1
        self.env_core.env.env.env.game_cnt = self.game_cnt
        self.action_masks_dict = dict(zip(self.env_core.agents, [np.ones(i[0].n) for i in self.action_dim]))
        if self.switch_seats:
            self.seats_permutation_idx += 1
            self.player_seats = dict(zip(list(self.player_seats.keys()),
                                         self.seats_permutation[self.seats_permutation_idx]))
            self.env_core.env.env.env.player_seats = self.player_seats
            if self.same_hand:
                self.env_core.seed(self.seed)

        self.env_core.reset()
        self.env_core.env.env.env.action_record = {}
        obs, _, _, _ = self.env_core.last()
        self.current_state = obs
        self.action_masks_dict[self.env_core.agent_selection] = obs['action_mask']
        self.all_observes = self.get_all_observes()
        self.init_info = self.get_info_after()
        return self.all_observes

    def step(self, joint_action):
        self.step_cnt_episode += 1
        joint_action=self.is_valid_action(joint_action)             #if illegal action exists, switch to random policy
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
            seats_payoff = np.array(self.env_core.env.env.env.env.get_payoffs())
            seats_player = dict(zip(self.player_seats.values(), self.player_seats.keys()))
            for seat in range(len(seats_payoff)):
                player_id = seats_player[seat]
                self.payoff[player_id] += seats_payoff[seat]
            self.env_core.env.env.env.payoff = self.payoff
            # self.env_core.render()

            if self.episode_count > 0:
                self.all_observes = self.reset_per_episode()
                info_after = self.init_info
        done = self.is_terminal()
        self.set_n_return()
        if done:
            print(f"Final payoff = {self.payoff}")
        return self.all_observes, reward, done, info_before, info_after

    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:
            raise Exception("Input joint action dimension should be {}, not {}.".format(
                self.n_player, len(joint_action)))

        current_seat_id = self.player_id_map[self.env_core.agent_selection]       #actually the seat number

        if not self.switch_seats:
            if (self.env_core.agent_selection in self.env_core.agents) and \
                    (not self.env_core.dones[self.env_core.agent_selection]):
                if joint_action[current_seat_id] is None or joint_action[current_seat_id][0] is None:
                    raise Exception("Action of current player is needed. Current player is {}, {}".format(
                        current_seat_id, self.env_core.agent_selection))

            for i in range(self.n_player):
                if joint_action[i] is None or joint_action[i][0] is None:
                    continue
                if len(joint_action[i][0]) != self.joint_action_space[i][0].n:
                    raise Exception("The input action dimension for player {} should be {}, not {}.".format(
                        i, self.joint_action_space[i][0].n, len(joint_action[i][0])))
        else:
            seats_player = dict(zip(self.player_seats.values(), self.player_seats.keys()))
            current_player = seats_player[current_seat_id]
            current_idx = int(current_player[-1])
            if len(joint_action[current_idx][0])!=self.joint_action_space[current_seat_id][0].n:
                raise Exception("The input action dimension for player {} should be {}, not {}.".format(
                    current_seat_id, self.joint_action_space[current_seat_id][0].n,
                    len(joint_action[current_idx][0])))

            if (np.array(joint_action[current_idx][0]) * self.action_masks_dict[self.env_core.agent_selection]).sum() == 0:
                # raise Exception(f"The action of player {current_player} has illegal action, "
                #                 f"input action = {joint_action[current_idx][0]} but the legal action should be {self.action_masks_dict[self.env_core.agent_selection]}")
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
        # for player_key, player_reward in self.env_core.rewards.items():
        #     player_id = self.player_id_map[player_key]
        #     self.n_return[player_id] += player_reward
        self.n_return = list(self.payoff.values())

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
            if self.player_id_map[self.env_core.agent_selection] == i:
                each = {"obs": each_obs, "is_new_episode": is_new_episode,      #
                        "current_move_player": self.env_core.agent_selection,
                        "controlled_player_index": i, "controlled_player_name": player_name}
            else:
                each = {"obs": None, "is_new_episode": is_new_episode,
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
        # info_after = {}
        # for i in range(self.n_player):
        #     temp_info = self.env_core.env.env.env.env.game.get_state(i)
        #
        #     for action_index, action in enumerate(temp_info['legal_actions']):
        #         temp_info['legal_actions'][action_index] = str(action)
        #     temp_info['stage'] = str(temp_info['stage'])
        #     info_after[self.player_id_reverse_map[i]] = temp_info


        return info_after
