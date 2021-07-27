# -*- coding:utf-8  -*-
# 作者：zruizhi   
# 创建时间： 2020/7/21 10:24 上午
# 描述：
from env.simulators.gridgame import GridGame
import random
from env.obs_interfaces.observation import *
from utils.discrete import Discrete


class Reversi(GridGame, GridObservation):
    # 棋盘大小n（n为偶数，且4≤n≤26）
    def __init__(self, conf):

        colors = conf.get('colors', [(255, 255, 255), (0, 0, 0), (245, 245, 245)])
        super().__init__(conf, colors)
        # 1：黑子 2：白子 默认黑子先下
        self.current_state = [[[0] * self.cell_dim for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.chess_player = 1
        self.n = self.board_width
        if self.n % 2:
            raise Exception("棋盘大小n，n应为偶数，且4≤n≤26，当前棋盘大小为%d" % self.n)
        # 黑白棋手当前棋盘的位置信息
        self.black = {(self.n // 2 - 1, self.n // 2), (self.n // 2, self.n // 2 - 1)}
        self.white = {(self.n // 2, self.n // 2), (self.n // 2 - 1, self.n // 2 - 1)}
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        # 当前棋手可以落子的合法位置
        self.legal_position = self.legal_positions()

        # 四个初始棋子的摆放
        self.current_state[int(self.n / 2 - 1)][int(self.n / 2 - 1)][0] = 2
        self.current_state[int(self.n / 2 - 1)][int(self.n / 2)][0] = 1
        self.current_state[int(self.n / 2)][int(self.n / 2 - 1)][0] = 1
        self.current_state[int(self.n / 2)][int(self.n / 2)][0] = 2
        self.all_observes = self.get_all_observes()

        # 上一个玩家是否无子可落
        self.last = 0
        # 游戏是否结束
        self.done = 0
        # 玩家当前action的reward
        self.reward = [0] * self.n_player
        self.step_cnt = 1
        # 各玩家最终棋子数量
        self.won = {}
        self.input_dimension = self.board_width * self.board_height
        self.action_dim = self.get_action_dim()

    def reset(self):
        self.current_state = [[[0] * self.cell_dim for _ in range(self.board_width)] for _ in range(self.board_height)]
        # 四个初始棋子的摆放
        self.current_state[int(self.n / 2 - 1)][int(self.n / 2 - 1)][0] = 2
        self.current_state[int(self.n / 2 - 1)][int(self.n / 2)][0] = 1
        self.current_state[int(self.n / 2)][int(self.n / 2 - 1)][0] = 1
        self.current_state[int(self.n / 2)][int(self.n / 2)][0] = 2

        # 上一个玩家是否无子可落
        self.last = 0
        # 游戏是否结束
        self.done = 0
        # 玩家当前action的reward
        self.reward = [0] * self.n_player
        self.step_cnt = 1

        self.black = {(self.n // 2 - 1, self.n // 2), (self.n // 2, self.n // 2 - 1)}
        self.white = {(self.n // 2, self.n // 2), (self.n // 2 - 1, self.n // 2 - 1)}

        # 1：黑子 2：白子 默认黑子先下
        self.chess_player = 1
        self.all_observes = self.get_all_observes()

        return self.all_observes

    def get_grid_observation(self, current_state, player_id, info_before):
        return current_state

    def get_dict_observation(self, player_id):
        key_info = {1: self.black, 2: self.white, "chess_player_idx": player_id, 'board_width': self.board_width,
                    'board_height': self.board_height}

        return key_info

    def get_all_observes(self):
        self.all_observes = []
        for i in range(self.n_player):
            each_obs = self.get_dict_observation(i + 1)
            self.all_observes.append(each_obs)

        return self.all_observes

    def get_next_state(self, joint_action):
        not_valid = self.is_not_valid_action(joint_action)
        info_after = {}
        if not not_valid:
            cur_action = joint_action[self.chess_player - 1]
            # print("current_state", self.current_state)
            next_state = self.current_state
            self.all_observes = self.get_all_observes()
            if not self.legal_position.keys():
                if self.last == 0:
                    # print("当前棋手没有可选位置")
                    self.last = 1
                    if self.chess_player == 1:
                        self.chess_player = 2
                    else:
                        self.chess_player = 1
                elif self.last == 1:
                    # print("双方均没有可选位置")
                    self.done = 1
                return next_state, info_after
            # 一方无子可落，交换下棋人选后 新的一方有合法位置
            else:
                if self.last == 1:
                    self.last = 0

            x, y = self.decode(cur_action)
            p, reverse = self.check_at(x, y)
            if reverse:
                # info_after = {}
                info_after["action"] = p
                info_after["reverse_positions"] = reverse

                if self.chess_player == 1:
                    self.black = self.black | {p} | set(reverse)
                    self.white = self.white - set(reverse)
                    self.chess_player = 2
                else:
                    self.chess_player = 1
                    self.white = self.white | {p} | set(reverse)
                    self.black = self.black - set(reverse)

                for p in self.black:
                    next_state[p[0]][p[1]][0] = 1
                for p in self.white:
                    next_state[p[0]][p[1]][0] = 2

                self.step_cnt += 1

            # return next_state, str(info_after)
            self.all_observes = self.get_all_observes()
            return self.all_observes, info_after

    def step_before_info(self, info=''):
        info = "当前棋手:%d" % self.chess_player
        self.legal_position = self.legal_positions()
        # info += "\n可落子位置，及落子后反色的位置集合: %s" % str(self.legal_position)

        return info

    def set_action_space(self):
        action_space = [[Discrete(self.board_height), Discrete(self.board_width)] for _ in range(self.n_player)]
        # action_space = [[self.board_height, self.board_width] for _ in range(self.n_player)]
        return action_space

    def is_not_valid_action(self, all_action):
        not_valid = 0
        if len(all_action) != self.n_player:
            raise Exception("joint action 维度不正确！", len(all_action))

        for i in range(self.n_player):
            if len(all_action[i]) != 2 or len(all_action[i][0]) != self.board_width or len(
                    all_action[i][1]) != self.board_height:
                raise Exception("玩家%d joint action维度不正确！" % i, all_action[i])
        return not_valid

    def get_reward(self, joint_action):
        return self.reward

    def legal_positions(self):
        """
        :return: 字典 {bai玩家或者hei玩家可以落子位置：反转对手子的位置}
        """
        if self.chess_player == 1:
            players = self.black
            _players = self.white
        else:
            players = self.white
            _players = self.black

        empty = [(i, j) for i in range(self.n) for j in range(self.n)]
        empty = set(empty) - self.black - self.white

        p_players_list_in = {}  # 如果落在p，会有的夹在中间的反色子集合
        for p in empty:
            all_r_players_list_in = []  # 所有方向的反色夹在中间子的集合
            for r in self.directions:
                _players_list_in = []  # 某一方向夹在中间反色子的集合
                i = 1
                lst = []
                while 1:
                    x, y = p[0] + i * r[0], p[1] + i * r[1]
                    if (x, y) in _players:
                        lst.append((x, y))
                        i += 1
                        nx, ny = p[0] + i * r[0], p[1] + i * r[1]
                        if (nx, ny) in players:
                            _players_list_in += lst
                            break
                        if nx < 0 or nx > self.n - 1 or ny < 0 or ny > self.n - 1:
                            break
                    else:
                        break

                if _players_list_in:  # 如果这个方向有夹在中间的反色子
                    all_r_players_list_in += _players_list_in
            if all_r_players_list_in:  # 如果落在p，会夹在中间的反色子集合【】
                p_players_list_in[p] = all_r_players_list_in
        return p_players_list_in

    # 当前位置是否可以落子，不可以则随机生成一个合法位置
    def check_at(self, x, y):
        p = (x, y)
        if p not in self.legal_position.keys():
            # 如果玩家未下在正确位置，则游戏结束
            """
            print("当前位置不合法!")
            self.done = 1

            if self.chess_player == 1:
                print("游戏结束，获胜方：白棋")
            else:
                print("游戏结束，获胜方：黑棋")
            raise Exception("Invalid position!", p)
            """
            # 如果玩家未下在正确位置，则随机生成一个合法的位置
            p = random.choice(list(self.legal_position))
            # print("当前位置不合法，随机生成一个合法位置：%s" % str(p))
        return p, self.legal_position[p]

    def check_win(self):
        if len(self.black) + len(self.white) == self.n ** 2:
            self.done = 1
        self.won = {1: len(self.black), 2: len(self.white)}
        if len(self.black) == 0:
            self.reward[1] = 100
            self.n_return = [0, 1]
            return 2
        if len(self.white) == 0:
            self.reward[0] = 100
            self.n_return = [1, 0]
            return 1
        if self.done:
            if len(self.black) > len(self.white):
                self.reward[0] = 100
                self.n_return = [1, 0]
                return 1
            elif len(self.black) < len(self.white):
                self.reward[1] = 100
                self.n_return = [0, 1]
                return 2
            else:
                self.reward[0] = 50
                self.reward[1] = 50
                self.n_return = [0.5, 0.5]
                return 3
        return 0

    def is_terminal(self):
        if self.done or self.step_cnt > self.max_step:
            return True
        flg = self.check_win()
        if flg == 0:
            return False
        else:
            return True

    def draw_board(self):
        cols = [chr(i) for i in range(65, 65 + self.board_width)]
        s = ', '.join(cols)
        print('  ', s)
        for i in range(self.board_width):
            print(chr(i + 65), self.current_state[i])

    def get_terminal_actions(self):
        not_input_valid = 1
        while not_input_valid:
            print("请输入落子横纵坐标[0-%d]，空格隔开：" % (self.board_width - 1))
            cur = input()
            l = cur.split(" ")
            x = int(l[0])
            y = int(l[1])
            if x < 0 or x >= self.board_height or y < 0 or y >= self.board_width:
                print("坐标超出限制，请重新输入!")
            else:
                self.last = 0
                return self.encode(x, y)

    def encode(self, x, y):
        joint_action = self.init_action_space()
        joint_action[self.chess_player - 1][0][x] = 1
        joint_action[self.chess_player - 1][1][y] = 1

        return joint_action

    def decode(self, each_action):
        x = each_action[0].index(1)
        y = each_action[1].index(1)
        return x, y

    def get_action_dim(self):
        action_dim = 1
        for i in range(len(self.joint_action_space[0])):
            action_dim *= self.joint_action_space[0][i].n

        return action_dim
