# -*- coding:utf-8  -*-
# 作者：zruizhi   
# 创建时间： 2020/7/10 10:24 上午   
# 描述：
from random import randrange
from env.simulators.gridgame import GridGame
from env.obs_interfaces.observation import *
from utils.discrete import Discrete


class GoBang(GridGame, GridObservation):
    def __init__(self, conf):
        colors = conf.get('colors', [(255, 255, 255), (0, 0, 0), (245, 245, 245)])

        super().__init__(conf, colors)
        if self.board_width != 15 or self.board_height != 15:
            raise Exception("棋盘大小应设置为15,15,当前棋盘大小为：%d,%d" % (self.board_width, self.board_height))
        self.current_state = [[[0] * self.cell_dim for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.all_observes = self.get_all_observes()
        # 1：黑子 2：白子 默认黑子先下
        self.chess_player = 1
        # 存储赢方位置信息
        self.won = {}
        self.step_cnt = 1

        # all_grids用于存储所有未被占用的格点
        self.all_grids = []
        for i in range(0, self.board_height):
            for j in range(0, self.board_width):
                self.all_grids.append((i, j))

        self.input_dimension = self.board_width * self.board_height
        self.action_dim = self.get_action_dim()

    def reset(self):
        self.current_state = [[[0] * self.cell_dim for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.chess_player = 1
        self.won = {}
        self.step_cnt = 1
        self.all_grids.clear()
        for i in range(0, self.board_height):
            for j in range(0, self.board_width):
                self.all_grids.append((i, j))
        self.all_observes = self.get_all_observes()

        return self.all_observes

    def set_action_space(self):
        action_space = [[Discrete(self.board_height), Discrete(self.board_width)] for _ in range(self.n_player)]
        # action_space = [[self.board_height, self.board_width] for _ in range(self.n_player)]
        return action_space

    def get_next_state(self, joint_action):
        info_after = {}
        not_valid = self.is_not_valid_action(joint_action)
        if not not_valid:
            next_state = self.current_state
            cur_action = joint_action[self.chess_player-1]

            x, y = self.decode(cur_action)
            if self.check_at(x, y):
                next_state[x][y][0] = self.chess_player
                if self.chess_player == 1:
                    self.chess_player = 2
                else:
                    self.chess_player = 1
                self.step_cnt += 1
                self.all_grids.remove((x, y))
            else:
                if len(self.all_grids) > 0:
                    pos = randrange(len(self.all_grids))
                    x, y = self.all_grids[pos]

                    next_state[x][y][0] = self.chess_player
                    if self.chess_player == 1:
                        self.chess_player = 2
                    else:
                        self.chess_player = 1
                    self.step_cnt += 1
                    self.all_grids.remove((x, y))
                else:
                    info_after = "棋盘已满"
            info_after["action"] = (x, y)
            self.all_observes = self.get_all_observes()

            return self.all_observes, info_after

    def step_before_info(self, info=''):
        info = "当前棋手:%d" % self.chess_player
        return info

    def is_not_valid_action(self, all_action):
        not_valid = 0
        if len(all_action) != self.n_player:
            raise Exception("joint action 维度不正确！", len(all_action))

        for i in range(self.n_player):
            if len(all_action[i]) != 2 or len(all_action[i][0]) != self.board_width or len(all_action[i][1]) != self.board_height:
                raise Exception("玩家%d joint action维度不正确！" % i, all_action[i])
        return not_valid

    def get_reward(self, joint_action):
        r = [0]*self.n_player
        if self.is_terminal():
            r[2-self.chess_player] = 100
            self.n_return[2-self.chess_player] = 1
        return r

    def encode(self, x, y):
        joint_action = self.init_action_space()
        joint_action[self.chess_player - 1][0][x] = 1
        joint_action[self.chess_player - 1][1][y] = 1

        return joint_action

    def decode(self, each_action):
        x = each_action[0].index(1)
        y = each_action[1].index(1)
        return x, y

    def get(self, row, col):
        if row < 0 or row >= self.board_height or col < 0 or col >= self.board_width:
            return 0
        return self.current_state[row][col][0]

    # 当前位置是否可以落子
    def check_at(self, x, y):
        if self.current_state[x][y][0] != 0:
            return False
        return True

    def check_win(self):
        dirs = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(self.board_width):
            for j in range(self.board_height):
                if self.current_state[i][j][0] == 0: continue
                id = self.current_state[i][j][0]
                for d in dirs:
                    x, y = i, j
                    count = 0
                    for k in range(5):
                        if self.get(x, y) != id: break
                        x += d[0]
                        y += d[1]
                        count += 1
                    if count == 5:
                        self.won = []
                        r, c = i, j
                        for z in range(5):
                            self.won.append([r, c])
                            r += d[0]
                            c += d[1]
                        return id
        return 0

    def is_terminal(self):
        flg = self.check_win()
        if self.step_cnt > self.max_step:
            return True
        if flg == 0:
            return False
        else:
            return True

    def get_grid_observation(self, current_state, player_id, info_before):
        return current_state

    def get_dict_observation(self, player_id):
        key_info = {"state_map": self.current_state, "chess_player_idx": player_id, 'board_width': self.board_width,
                    'board_height': self.board_height}

        return key_info

    def get_all_observes(self):
        self.all_observes = []
        for i in range(self.n_player):
            each_obs = self.get_dict_observation(i + 1)
            self.all_observes.append(each_obs)

        return self.all_observes

    def get_terminal_actions(self):
        not_input_valid = 1
        while not_input_valid:
            print("请输入落子横纵坐标[0-%d]，空格隔开：" % (self.board_width-1))
            cur = input()
            l = cur.split(" ")
            x = int(l[0])
            y = int(l[1])
            if x < 0 or x >= self.board_height or y < 0 or y >= self.board_width:
                print("坐标超出限制，请重新输入!")
            else:
                return self.encode(x, y)

    def get_action_dim(self):
        action_dim = 1
        for i in range(len(self.joint_action_space[0])):
            action_dim *= self.joint_action_space[0][i].n

        return action_dim




