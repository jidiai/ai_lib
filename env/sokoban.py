# -*- coding:utf-8  -*-
# 作者：zruizhi   
# 创建时间： 2020/7/30 17:24 下午
# 描述：
from env.simulators.gridgame import GridGame
import random
from env.obs_interfaces.observation import *
from utils.discrete import Discrete

levels = {
    1: [
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 2, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 3, 5, 3, 2, 1],
        [1, 2, 0, 3, 4, 1, 1, 1],
        [1, 1, 1, 1, 3, 1, 0, 0],
        [0, 0, 0, 1, 2, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0]
    ],
    2: [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 2, 2, 1, 0, 0],
        [0, 1, 1, 0, 2, 1, 1, 0],
        [0, 1, 0, 5, 3, 2, 1, 0],
        [1, 1, 0, 3, 4, 0, 1, 1],
        [1, 0, 0, 1, 3, 3, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]
}


class Sokoban(GridGame, GridObservation):
    def __init__(self, conf):
        colors = conf.get('colors', [(255, 255, 255), (0, 0, 0), (255, 69, 0), (222, 184, 135)])
        super().__init__(conf, colors)
        # 0：没有 1：围墙 2：目标点 3：箱子 4-n_player+3：人物
        self.n_cell_type = self.n_player + 3
        self.step_cnt = 1
        level = int(conf['level'])
        self.map = levels[level]

        # 方向[0, 1, 2, 3]分别表示[上，下，左，右]
        self.actions_name = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.walls = []
        self.boxes = []
        self.targets = []
        self.players = []
        self.current_state = self.init_state
        self.all_observes = self.get_all_observes()
        self.won = {'total boxes': len(self.boxes)}
        self.success_box_each_step = 0
        self.input_dimension = self.board_width * self.board_height
        self.action_dim = self.get_action_dim()

    def reset(self):
        self.step_cnt = 1
        self.walls = []
        self.boxes = []
        self.targets = []
        self.players = []
        self.current_state = self.init_state
        self.all_observes = self.get_all_observes()

        return self.all_observes

    def check_win(self):
        cnt = 0
        for box in self.boxes:
            for t in self.targets:
                if box.row == t[0] and box.col == t[1]:
                    cnt += 1
        self.won['success boxes'] = cnt
        return cnt

    def set_action_space(self):
        action_space = [[Discrete(4)] for _ in range(self.n_player)]
        # action_space = [[4] for _ in range(self.n_player)]
        return action_space

    @property
    def init_state(self):
        b_cnt = 0
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == 1:
                    self.walls.append([i, j])
                elif self.map[i][j] == 2:
                    self.targets.append([i, j])
                elif self.map[i][j] == 3:
                    box_obj = GameObject('b' + str(b_cnt), i, j, 0)
                    self.boxes.append(box_obj)
                    b_cnt += 1
                else:
                    for p in range(4, self.n_cell_type+1):
                        if self.map[i][j] == p:
                            player = GameObject(p, i, j, 0)
                            self.players.append(player)
        # self.players.sort(key=lambda pl: pl.object_id)
        current_state = [[[0] * self.cell_dim for _ in range(self.board_width)] for _ in range(self.board_height)]
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                current_state[i][j][0] = self.map[i][j]
        self.init_info = {"walls": self.walls, "targets": self.targets,
                          "players_position": [[p.row, p.col] for p in self.players],
                          "boxes_position": [[b.row, b.col] for b in self.boxes]}
        return current_state

    def update_state(self):
        next_state = [[[0] * self.cell_dim for _ in range(self.board_width)] for _ in range(self.board_height)]

        for pos in self.walls:
            next_state[pos[0]][pos[1]][0] = 1

        for pos in self.targets:
            next_state[pos[0]][pos[1]][0] = 2
        # box 可能会覆盖 target
        for box in self.boxes:
            next_state[box.row][box.col][0] = 3
        # player 可能会覆盖 target
        for player in self.players:
            next_state[player.row][player.col][0] = player.object_id

        return next_state

    def out_of_board(self, x, y):
        if x <= 0 or x >= self.board_height:
            return True
        if y <= 0 or y >= self.board_width:
            return True

        return False

    def get_next_state(self, joint_action):
        not_valid = self.is_not_valid_action(joint_action)
        self.success_box_each_step = 0
        if not not_valid:
            origin_box = self.check_win()
            # print("current_state", self.current_state)
            random.shuffle(self.players)
            # print("after shuffle", [i.object_id for i in self.players])
            unprocessed_players = self.players
            occupied_pos = []
            # 存在死锁情况
            loop_cnt = 0
            # 各玩家行动
            while unprocessed_players and loop_cnt < 10:
                nxt_process = []
                for i in range(len(unprocessed_players)):
                    p = unprocessed_players[i]
                    act_idx = p.object_id-4
                    p.direction = joint_action[act_idx][0].index(1)
                    # print("%d direction" % p.object_id, self.actions_name[p.direction])

                    p1 = p.get_next_pos(p.row, p.col)
                    if self.out_of_board(p1[0], p1[1]):
                        continue

                    p2 = p.get_next_two_pos()

                    # 位置已被其他玩家或者箱子占领
                    if p1 in occupied_pos or (self.current_state[p1[0]][p1[1]][0] == 3 and p2 in occupied_pos):
                        continue
                    # 位置不合法
                    if self.is_valid_pos(p1, p2) == 0:
                        continue
                    elif self.is_valid_pos(p1, p2) == 1:
                        occupied_pos.append(p1)
                        # 如果玩家和箱子一起移动
                        if self.current_state[p1[0]][p1[1]][0] == 3:
                            occupied_pos.append(p2)
                            for box in self.boxes:
                                if box.row == p1[0] and box.col == p1[1]:
                                    box.direction = p.direction
                                    box.move()
                                    break
                        p.move()
                        # 更新状态
                        self.current_state = self.update_state()
                    # 需要等待其他玩家行动后，待确认的玩家
                    else:
                        nxt_process.append(p)

                unprocessed_players = nxt_process
                loop_cnt += 1

            # 此轮成功推到目的地的箱子数
            self.success_box_each_step = self.check_win() - origin_box

            # 更新状态
            next_state = self.update_state()
            self.current_state = next_state
            self.step_cnt += 1

            players = sorted(self.players, key=lambda pl: pl.object_id)
            info_after = {"players_position": [[p.row, p.col] for p in players],
                          "boxes_position": [[b.row, b.col] for b in self.boxes]}
            self.all_observes = self.get_all_observes()

            return self.all_observes, info_after

    def get_grid_observation(self, current_state, player_id, info_before):
        return current_state

    def get_dict_observation(self, player_id):
        key_info = {"state_map": self.current_state, "player_idx": player_id, 'board_width': self.board_width,
                    'board_height': self.board_height}

        return key_info

    def get_all_observes(self):
        self.all_observes = []
        for i in range(self.n_player):
            each_obs = self.get_dict_observation(i + 4)
            self.all_observes.append(each_obs)

        return self.all_observes

    def is_not_valid_action(self, all_action):
        not_valid = 0
        if len(all_action) != self.n_player:
            raise Exception("joint action 维度不正确！", len(all_action))

        for i in range(self.n_player):
            if len(all_action[i][0]) != 4:
                raise Exception("玩家%d joint action维度不正确！" % i, all_action[i])
        return not_valid

    def is_valid_pos(self, p1, p2):
        # 前面是空的或者是目标位置
        if self.current_state[p1[0]][p1[1]][0] in (0, 2):
            return 1
        # 前面是墙
        elif self.current_state[p1[0]][p1[1]][0] == 1:
            return 0
        # 前面是箱子
        elif self.current_state[p1[0]][p1[1]][0] == 3:
            # 越界
            if self.out_of_board(p2[0], p2[1]):
                return 0
            # 箱子前面是空的或者是目标位置
            if self.current_state[p2[0]][p2[1]][0] in (0, 2):
                return 1
            # 箱子前面是墙或者箱子
            elif self.current_state[p2[0]][p2[1]][0] in (1, 3):
                return 0
            else:
                return self.current_state[p2[0]][p2[1]][0]
        # 前面是玩家，需要判断此轮结算是否移动出了空位
        else:
            return self.current_state[p1[0]][p1[1]][0]

    def get_reward(self, joint_action):
        # print("success_box_each_step", self.success_box_each_step)
        # r = [self.success_box_each_step * self.max_step // (self.step_cnt-1)] * self.n_player
        step_reward = self.success_box_each_step - 1
        r = [step_reward] * self.n_player
        self.n_return[0] += step_reward
        self.n_return = [self.n_return[0]] * self.n_player
        return r

    def is_terminal(self):
        return self.step_cnt > self.max_step or self.check_win() == len(self.boxes)

    def encode(self, actions):
        joint_action = self.init_action_space()
        if len(actions) != self.n_player:
            raise Exception("action输入维度不正确！", len(actions))
        for i in range(self.n_player):
            joint_action[i][0][int(actions[i])] = 1
        return joint_action

    def get_terminal_actions(self):
        print("请输入%d个玩家的动作方向[0-3](上下左右)，空格隔开：" % self.n_player)
        cur = input()
        actions = cur.split(" ")
        return self.encode(actions)

    def get_action_dim(self):
        action_dim = 1
        for i in range(len(self.joint_action_space[0])):
            action_dim *= self.joint_action_space[0][i].n

        return action_dim


class GameObject():
    def __init__(self, object_id, row, col, direction):
        self.object_id = object_id
        self.row = row
        self.col = col
        self.direction = direction

    def move(self):
        next_pos = self.get_next_pos(self.row, self.col)
        self.row = next_pos[0]
        self.col = next_pos[1]

    def get_next_pos(self, cur_row, cur_col):
        nxt_row = cur_row
        nxt_col = cur_col

        if self.direction == 0:
            nxt_row = cur_row - 1
        elif self.direction == 1:
            nxt_row = cur_row + 1
        elif self.direction == 2:
            nxt_col = cur_col - 1
        elif self.direction == 3:
            nxt_col = cur_col + 1

        return [nxt_row, nxt_col]

    def get_next_two_pos(self):
        next_pos = self.get_next_pos(self.row, self.col)
        next_two_pos = self.get_next_pos(next_pos[0], next_pos[1])

        return next_two_pos








