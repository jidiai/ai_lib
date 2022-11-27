# -*- coding:utf-8  -*-
# Time  : 2022/3/22 下午4:09
# Author: Yahui Cui
import copy
import json
import os
from pathlib import Path

from env.simulators.game import Game
from env.obs_interfaces.observation import DictObservation
import numpy as np
from utils.box import Box
import random


import sys
import os.path as osp
import pygame
from pygame.locals import *
import igraph
from scipy.spatial import distance


current_dir = str(Path(__file__).resolve().parent)


def load_config(file_path):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config


class LogisticsEnv2(Game, DictObservation):
    def __init__(self, conf):
        super().__init__(
            conf["n_player"],
            conf["is_obs_continuous"],
            conf["is_act_continuous"],
            conf["game_name"],
            conf["agent_nums"],
            conf["obs_type"],
        )
        map_path = os.path.join(current_dir, "logistics2", "map_0.json")
        map_conf = load_config(map_path)
        self.map_conf = map_conf
        self.n_goods = int(map_conf["n_goods"])
        self.max_step = int(conf["max_step"])

        self.players = []
        self.init_map()  # 根据地图数据初始化地图

        self.step_cnt = 0
        self.n_return = [0] * self.n_player
        self.current_state = None
        # 每个玩家的action space list, 可以根据player_id获取对应的single_action_space
        self.joint_action_space = self.set_action_space()
        self.info = {}
        self.won = {}
        self.init_info = None
        self.viewer = None
        self.all_observes = self.reset()

    def init_map(self):
        goods_list = []
        for goods_info in self.map_conf["goods"]:
            goods = Goods(goods_info)
            goods_list.append(goods)

        # 添加图中的节点
        vertices = copy.deepcopy(self.map_conf["vertices"])
        for vertex_info in vertices:
            vertex_info.update({"goods": goods_list})
            vertex = Vertex(vertex_info["key"], vertex_info)
            self.players.append(vertex)

        # 添加图中的边
        roads = copy.deepcopy(self.map_conf["roads"])
        for road_info in roads:
            start = road_info["start"]
            end = road_info["end"]
            road = Road(road_info)
            self.players[start].add_neighbor(end, road)
            if not self.map_conf["is_graph_directed"]:  # 若是无向图，则加上反方向的边
                self.players[end].add_neighbor(start, road)

        # 初始化每个节点
        for i in range(self.n_player):
            self.players[i].update_init_storage()

    def reset(self):
        self.players = []
        self.init_map()

        self.step_cnt = 0
        self.n_return = [0] * self.n_player

        self.current_state = self.get_current_state()
        self.all_observes = self.get_all_observations()
        self.won = {}

        self.info = {
            "productions": [self.players[i].production for i in range(self.n_player)],
            "upper_volume": [
                self.players[i].upper_capacity for i in range(self.n_player)
            ],
            "upper_capacity": [
                [act.high.tolist() for act in v_action]
                for v_action in self.joint_action_space
            ],
        }

        return self.all_observes

    def step(self, all_actions):
        self.step_cnt += 1
        all_actions = self.bound_actions(all_actions)

        self.current_state = self.get_next_state(all_actions)
        self.all_observes = self.get_all_observations()

        reward, single_rewards = self.get_reward(all_actions)
        done = self.is_terminal()
        self.info.update(
            {"actual_actions": all_actions, "single_rewards": single_rewards}
        )

        return self.all_observes, reward, done, "", self.info

    def bound_actions(self, all_actions):  # 对每个节点的动作进行约束
        bounded_actions = []
        for i in range(self.n_player):
            result = self.players[i].bound_actions(all_actions[i])
            bounded_actions.append(result)

        return bounded_actions

    def get_current_state(self):
        current_state = []
        for i in range(self.n_player):
            current_state.append(self.players[i].init_storage.copy())

        return current_state

    def get_next_state(self, all_actions):
        assert len(all_actions) == self.n_player

        # 统计每个节点当天运出的货物量out_storages，以及接收的货物量in_storages
        out_storages = np.zeros((self.n_player, self.n_goods))
        in_storages = np.zeros((self.n_player, self.n_goods))
        for i in range(self.n_player):
            action = np.array(all_actions[i])
            out_storages[i] = np.sum(action, axis=0)
            connections = self.players[i].get_connections()
            for idx, nbr in enumerate(connections):
                in_storages[nbr] += action[idx]

        # 更新每个节点当天的最终库存量以及下一天的初始库存量，
        # 并记录每个节点当天最开始的初始库存start_storages和消耗量demands，用于可视化
        next_state = []
        start_storages, demands = [], []
        for i in range(self.n_player):
            start_storages.append(self.players[i].final_storage.copy())
            demands.append(self.players[i].demands)

            self.players[i].update_final_storage(out_storages[i], in_storages[i])
            self.players[i].update_init_storage()
            next_state.append(self.players[i].init_storage.copy())

        self.info.update({"start_storages": start_storages, "demands": demands})

        return next_state

    def get_dict_observation(self, current_state, player_id, info_before):
        obs = {
            "obs": current_state,
            "connected_player_index": self.players[player_id].get_connections(),
            "controlled_player_index": player_id,
        }
        return obs

    def get_all_observations(self, info_before=""):
        all_obs = self.get_dict_many_observation(
            self.current_state, range(self.n_player), info_before
        )
        return all_obs

    def get_reward(self, all_actions):
        total_reward = 0
        single_rewards = []
        for i in range(self.n_player):
            action = all_actions[i]
            reward = self.players[i].calc_reward(action)
            total_reward += reward
            single_rewards.append(reward)
            self.n_return[i] += reward

        return total_reward, single_rewards

    def set_action_space(self):
        action_space = []

        for i in range(self.n_player):
            vertex = self.players[i]
            action_space_i = []
            for j in vertex.get_connections():
                road = vertex.get_road(j)
                high = [
                    road.upper_capacity // vertex.goods[k].volume
                    for k in range(self.n_goods)
                ]
                space = Box(np.zeros(self.n_goods), np.array(high), dtype=np.float64)
                action_space_i.append(space)
            action_space.append(action_space_i)

        return action_space

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def is_terminal(self):
        is_done = self.step_cnt >= self.max_step
        return is_done

    def get_render_data(self, current_state=None):
        render_data = {
            "day": self.step_cnt,
            "storages": self.info["start_storages"],
            "productions": self.info["productions"],
            "demands": self.info["demands"],
            "total_reward": sum(self.n_return),
            "single_rewards": self.info["single_rewards"],
            "actions": self.info["actual_actions"],
        }

        return render_data

    def render(self):
        render_data = self.get_render_data()
        if self.viewer is None:
            pd_gaps, p_gaps, all_connections, all_times = [], [], [], []
            for i in range(self.n_player):
                vertex = self.players[i]
                pd_gaps.append(
                    sum([p - d for p, d in zip(vertex.production, vertex.lambda_)])
                )
                connections = vertex.get_connections()
                all_connections.append(connections)
                times = [vertex.get_road(j).trans_time for j in connections]
                all_times.append(times)
            for j in range(self.n_player):
                s_gaps = render_data["storages"][j]
                p_gaps.append(sum(p for p in s_gaps))
            network_data = {
                "n_vertex": self.n_player,
                "is_abs_coords": self.map_conf.get("is_abs_coords"),
                "v_coords": self.map_conf.get("coords"),
                "connections": all_connections,
                "pd_gaps": pd_gaps,
                "p_gaps": p_gaps,
                "trans_times": all_times,
            }
            self.viewer = Viewer(1200, 800, network_data)

        self.viewer.render(render_data)

    def check_win(self):
        return "-1"


class Vertex(object):
    def __init__(self, key, info):
        self.key = key
        self.connectedTo = {}

        self.goods = info["goods"]
        self.n_goods = len(self.goods)
        self.production = info["production"]
        self.init_storage = [0] * self.n_goods
        self.final_storage = info["init_storage"]
        self.upper_capacity = info["upper_capacity"]

        self.store_cost = info["store_cost"]
        self.loss_cost = info["loss_cost"]
        self.storage_loss = [0] * self.n_goods  # 更新完当天的最终库存量后，统计当天的库存溢出量
        self.init_storage_loss = [0] * self.n_goods  # 因为每次状态更新会提前计算下一天的初始库存量，
        # 若不单独记录初始库存的溢出量，则会在计算每日reward时出错
        self.lambda_ = info["lambda"]

        self.demands = None
        self.fulfillment = None

    def add_neighbor(self, nbr, road):
        self.connectedTo.update({nbr: road})

    def get_connections(self):
        return list(self.connectedTo.keys())

    def get_road(self, nbr):
        return self.connectedTo.get(nbr)

    def get_demands(self):
        demands = [
            np.random.poisson(lam=self.lambda_[k], size=1)[0]
            for k in range(self.n_goods)
        ]
        return demands

    def bound_actions(self, actions):
        if len(actions) == 0:  # 若该节点没有动作，则不做约束
            return actions

        actions = np.array(actions)
        result = np.zeros_like(actions)
        for k in range(self.n_goods):
            actual_trans = np.sum(actions[:, k])
            if self.init_storage[k] > 0:
                if actual_trans > self.init_storage[k]:  # 运出的总货物量超过初始库存量
                    # 所有道路上同种货物量进行等比例缩放
                    result[:, k] = actions[:, k] * self.init_storage[k] / actual_trans
                else:  # 合法动作
                    result[:, k] = actions[:, k]

        goods_volumes = np.array([goods.volume for goods in self.goods])
        total_volumes = np.dot(result, goods_volumes.T)  # 每条道路上运输的总体积
        for i, nbr in enumerate(self.get_connections()):
            road = self.get_road(nbr)
            if total_volumes[i] > road.upper_capacity:  # 该道路上货物总体积超过道路容量
                # 对该道路上每种货物，按其体积占比，对数量进行等比例缩放
                result[i] = result[i] * road.upper_capacity / total_volumes[i]

        return result.tolist()

    def update_init_storage(self):
        self.demands = self.get_demands()
        total_volume = 0
        for k in range(self.n_goods):
            self.init_storage[k] = (
                self.final_storage[k] - self.demands[k] + self.production[k]
            )
            if self.init_storage[k] > 0:
                total_volume += self.init_storage[k] * self.goods[k].volume

        self.init_storage_loss = [0] * self.n_goods
        while total_volume > self.upper_capacity:  # 当天初始库存超过存储容量上限
            key = random.randint(0, self.n_goods - 1)
            if self.init_storage[key] > 0:
                self.init_storage[key] -= 1
                self.init_storage_loss[key] += 1
                total_volume -= self.goods[key].volume

    def update_final_storage(self, out_storage, in_storage):
        self.fulfillment = self.demands.copy()
        total_volume = 0
        for k in range(self.n_goods):
            self.fulfillment[k] -= min(0, self.final_storage[k])
            self.final_storage[k] = (
                self.init_storage[k] - out_storage[k] + in_storage[k]
            )
            if self.final_storage[k] > 0:
                total_volume += self.final_storage[k] * self.goods[k].volume

        self.storage_loss = self.init_storage_loss
        while total_volume > self.upper_capacity:  # 当天最终库存超过存储容量上限
            key = random.randint(0, self.n_goods - 1)
            if self.final_storage[key] > 0:
                self.final_storage[key] -= 1
                self.storage_loss[key] += 1
                total_volume -= self.goods[key].volume

        for k in range(self.n_goods):
            self.fulfillment[k] += min(0, self.final_storage[k])

    def calc_reward(self, actions, mu=1, scale=100):
        connections = self.get_connections()
        assert len(actions) == len(connections)

        reward = 0
        for k in range(self.n_goods):
            goods = self.goods[k]
            # 1. 需求满足回报
            reward += goods.price * self.fulfillment[k]
            # 2. 货物存储成本（兼惩罚项）
            if self.final_storage[k] >= 0:
                reward -= self.store_cost * self.final_storage[k] * goods.volume
            else:
                reward += mu * self.final_storage[k] * goods.volume
            # 3. 舍弃货物损失
            reward -= self.loss_cost[k] * self.storage_loss[k]

        # 4. 货物运输成本
        for (action, nbr) in zip(actions, connections):
            road = self.get_road(nbr)
            volume = 0
            for k in range(self.n_goods):
                volume += action[k] * self.goods[k].volume
            reward -= road.trans_cost * road.trans_time * volume

        return reward / scale


class Road(object):
    def __init__(self, info):
        self.upper_capacity = info["upper_capacity"]
        self.trans_time = info["trans_time"]
        self.trans_cost = info["trans_cost"]


class Goods(object):
    def __init__(self, info):
        self.volume = info["volume"]
        self.price = info["price"]


resource_path = os.path.join(current_dir, "logistics2", "resources")

# NOTE: FPS*SPD应为24的倍数，否则可能导致货车到达终点时偏移仓库图标中心
FPS = 60  # Frame Per Second，帧率，即每秒播放的帧数
SPD = 4  # Second Per Day，游戏中每天所占的秒数
prov_map = {
    0: "北京",
    1: "天津",
    2: "河北",
    3: "山西",
    4: "内蒙古",
    5: "辽宁",
    6: "吉林",
    7: "黑龙江",
    8: "上海",
    9: "江苏",
    10: "浙江",
    11: "安徽",
    12: "福建",
    13: "江西",
    14: "山东",
    15: "河南",
    16: "湖北",
    17: "湖南",
    18: "广东",
    19: "广西",
    20: "海南",
    21: "重庆",
    22: "四川",
    23: "贵州",
    24: "云南",
    25: "西藏",
    26: "陕西",
    27: "甘肃",
    28: "青海",
    29: "宁夏",
    30: "新疆",
}


class Viewer(object):
    def __init__(self, width, height, network_data):
        self.width = width
        self.height = height
        self.v_radius = 42
        self.n_vertex = network_data["n_vertex"]
        self.pd_gaps = network_data["pd_gaps"]  # 每个节点生产量和平均消耗量之间的差距
        self.p_gaps = network_data["p_gaps"]
        self.connections = network_data["connections"]
        self.trans_times = network_data["trans_times"]
        if network_data["is_abs_coords"]:  # 若提供了绝对坐标，则直接采用
            self.v_coords = network_data["v_coords"]
        else:  # 若未提供坐标，或是相对坐标，则自动计算坐标
            self.v_coords = self._spread_vertex(network_data["v_coords"])

        pygame.init()
        pygame.display.set_caption("Simple Logistics Simulator")
        self.screen = pygame.display.set_mode([width, height])
        self.screen.fill("white")
        self.FPSClock = pygame.time.Clock()
        self.is_paused = False

        self.font1 = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 24)
        self.font2 = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 18)
        self.font3 = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 14)
        self.p_img = pygame.image.load(
            osp.join(resource_path, "img/produce.png")
        ).convert_alpha()
        self.p_img = pygame.transform.scale(self.p_img, (16, 16))
        self.d_img = pygame.image.load(
            osp.join(resource_path, "img/demand.png")
        ).convert_alpha()
        self.d_img = pygame.transform.scale(self.d_img, (14, 14))
        self.pause = pygame.image.load(
            osp.join(resource_path, "img/pause.png")
        ).convert_alpha()
        self.play = pygame.image.load(
            osp.join(resource_path, "img/play.png")
        ).convert_alpha()
        # 设置暂停按钮位置
        self.pause_rect = self.pause.get_rect()
        self.pause_rect.right, self.pause_rect.top = self.width - 10, 10

        self.background = self.init_background()
        self.warehouses = self.init_warehouses()
        self.trucks = self.init_trucks()

    def init_background(self):
        # 导入背景地图图像
        bg_img = pygame.image.load(
            osp.join(resource_path, "img/china_map.png")
        ).convert_alpha()
        self.screen.blit(bg_img, (0, 0))

        # 绘制道路
        drawn_roads = []
        for i in range(self.n_vertex):
            start = self.v_coords[i]
            for j in self.connections[i]:
                if (j, i) in drawn_roads:
                    continue
                end = self.v_coords[j]
                self._draw_road(
                    start,
                    end,
                    width=5,
                    border_color=(252, 122, 90),
                    fill_color=(255, 172, 77),
                )
                drawn_roads.append((i, j))
        # 加入固定的提示
        self.add_notation()

        # 保存当前初始化的背景，便于后续刷新时使用
        background = self.screen.copy()
        return background

    def init_warehouses(self):
        warehouse_list = []
        for i in range(self.n_vertex):
            if self.p_gaps[i] > 0:
                warehouse = Warehouse(i, self.v_coords[i], (60, 179, 113, 1))
                warehouse_list.append(warehouse)
            elif self.p_gaps[i] < 0:
                warehouse = Warehouse(i, self.v_coords[i], (210, 77, 87, 1))
                warehouse_list.append(warehouse)
            elif self.p_gaps[i] == 0:
                warehouse = Warehouse(i, self.v_coords[i], (128, 128, 128, 1))
                warehouse_list.append(warehouse)
        return warehouse_list

    def init_trucks(self):
        trucks_list = []
        for i in range(self.n_vertex):
            start = self.v_coords[i]
            trucks = []
            for j, time in zip(self.connections[i], self.trans_times[i]):
                end = self.v_coords[j]
                truck = Truck((i, j), start, end, time)
                trucks.append(truck)
            trucks_list.append(trucks)

        return trucks_list

    def _spread_vertex(self, v_coords=None):
        if not v_coords:  # 若没有指定相对坐标，则随机将节点分布到画布上
            g = igraph.Graph()
            g.add_vertices(self.n_vertex)
            for i in range(self.n_vertex):
                for j in self.connections[i]:
                    g.add_edge(i, j)
            layout = g.layout_kamada_kawai()
            layout_coords = np.array(layout.coords).T
        else:  # 否则使用地图数据中指定的节点相对坐标
            layout_coords = np.array(v_coords).T

        # 将layout的坐标原点对齐到左上角
        layout_coords[0] = layout_coords[0] - layout_coords[0].min()
        layout_coords[1] = layout_coords[1] - layout_coords[1].min()

        # 将layout的坐标映射到画布坐标，并将图形整体居中
        stretch_rate = min(
            (self.width - 2 * self.v_radius - 240) / layout_coords[0].max(),
            (self.height - 2 * self.v_radius - 60) / layout_coords[1].max(),
        )
        # x方向左侧留出200，用于信息显示
        margin_x = (self.width - layout_coords[0].max() * stretch_rate) // 2 + 90
        margin_y = (self.height - layout_coords[1].max() * stretch_rate) // 2
        vertex_coord = []
        for i in range(self.n_vertex):
            x = margin_x + int(layout_coords[0, i] * stretch_rate)
            y = margin_y + int(layout_coords[1, i] * stretch_rate)
            vertex_coord.append((x, y))

        return vertex_coord

    def _draw_road(self, start, end, width, border_color=(0, 0, 0), fill_color=None):
        length = distance.euclidean(start, end)
        sin = (end[1] - start[1]) / length
        cos = (end[0] - start[0]) / length

        vertex = lambda e1, e2: (
            start[0] + (e1 * length * cos + e2 * width * sin) / 2,
            start[1] + (e1 * length * sin - e2 * width * cos) / 2,
        )
        vertices = [vertex(*e) for e in [(0, -1), (0, 1), (2, 1), (2, -1)]]

        if not fill_color:
            pygame.draw.polygon(self.screen, border_color, vertices, width=3)
        else:
            pygame.draw.polygon(self.screen, fill_color, vertices, width=0)
            pygame.draw.polygon(self.screen, border_color, vertices, width=2)

    def add_notation(self):
        text1 = self.font3.render("黑:库存量", True, (44, 44, 44), (255, 255, 255))
        self.screen.blit(text1, (18, 65))

        text2 = self.font3.render(":生产量", True, (35, 138, 32), (255, 255, 255))
        self.screen.blit(text2, (32, 85))
        self.screen.blit(self.p_img, (17, 85))

        text3 = self.font3.render(":消耗量", True, (251, 45, 45), (255, 255, 255))
        self.screen.blit(text3, (32, 105))
        self.screen.blit(self.d_img, (17, 105))

        text4 = self.font3.render("蓝:节点奖赏", True, (12, 140, 210), (255, 255, 255))
        self.screen.blit(text4, (18, 125))

        text5 = self.font3.render(
            f"tips:点击图中节点/运输车辆查看详细信息", True, (44, 44, 44), (255, 255, 255)
        )
        self.screen.blit(text5, (18, 142))

    def update(self, render_data):
        day = render_data["day"]
        storages = render_data["storages"]  # list per vertex
        productions = render_data["productions"]  # list per vertex
        demands = render_data["demands"]  # list per vertex
        total_reward = render_data["total_reward"]
        single_rewards = render_data["single_rewards"]
        actions = render_data["actions"]  # matrix per vertex

        self.screen.blit(self.background, (0, 0))
        day_text = self.font1.render(f"第{day}天", True, (44, 44, 44), (255, 255, 255))
        self.screen.blit(day_text, (18, 10))
        r_text = self.font2.render(
            f"累计奖赏:{round(total_reward, 2)}", True, (44, 44, 44), (255, 255, 255)
        )
        self.screen.blit(r_text, (18, 40))

        # 绘制每个仓库节点
        for warehouse, s, p, d, r in zip(
            self.warehouses, storages, productions, demands, single_rewards
        ):
            warehouse.update(s, p, d, r, self.screen)
            # warehouse.draw(self.screen, s)

        # 绘制每个action对应的货车
        for i in range(self.n_vertex):
            for truck, action in zip(self.trucks[i], actions[i]):
                truck.update(action)
                truck.draw(self.screen)

    def _check_click(self, pos):
        # 检测是否点到暂停按钮
        if self.pause_rect.collidepoint(pos):
            self.is_paused = not self.is_paused
            return

        if self.is_paused:
            # 检测是否点击到仓库节点
            for warehouse in self.warehouses:
                details = warehouse.click(pos)
                if details:
                    height = details.get_height()
                    self.screen.blit(details, (0, self.height - height))
                    return

            # 检测是否点击到货车
            for i in range(self.n_vertex):
                for truck in self.trucks[i]:
                    details = truck.click(pos)
                    if details:
                        height = details.get_height()
                        self.screen.blit(details, (0, self.height - height))
                        return

    def render(self, render_data):
        current_frame = 0
        while current_frame < FPS * SPD:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    pressed_array = pygame.mouse.get_pressed(3)
                    if pressed_array[0]:
                        pos = pygame.mouse.get_pos()
                        self._check_click(pos)

            if not self.is_paused:
                self.update(render_data)
                self.screen.blit(self.pause, self.pause_rect)
                current_frame += 1
            else:
                self.screen.fill("white", self.pause_rect)
                self.screen.blit(self.play, self.pause_rect)

            self.FPSClock.tick(FPS)
            pygame.display.update()


class Warehouse(object):
    def __init__(self, key, pos, color=(0, 0, 0), radius=15):
        self.key = key
        self.rect = Rect(0, 0, 2 * radius, 2 * radius)
        self.rect.center = pos
        self.radius = radius
        self.color = color
        self.fill_color = self._lighten_color(color, alpha=0.2)

        self.font = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 14)
        self.p_img = pygame.image.load(
            osp.join(resource_path, "img/produce.png")
        ).convert_alpha()
        self.p_img = pygame.transform.scale(self.p_img, (16, 16))
        self.d_img = pygame.image.load(
            osp.join(resource_path, "img/demand.png")
        ).convert_alpha()
        self.d_img = pygame.transform.scale(self.d_img, (14, 14))

        self.storage = None
        self.production = None
        self.demand = None
        self.reward = None
        self.details = pygame.Surface((200, 200))

    def update(self, storage, production, demand, reward, screen):
        self.storage = storage
        self.production = production
        self.demand = demand
        self.reward = reward
        sum_storages = sum(p for p in self.storage)
        if sum_storages > 0:
            self.draw(screen, (60, 179, 113))
        elif sum_storages < 0:
            self.draw(screen, (210, 77, 87))
        elif sum_storages == 0:
            self.draw(screen, (128, 128, 128))

    def draw(self, screen, color):
        circle_color = color
        circle_fill_color = self._lighten_color(circle_color, alpha=0.2)
        pygame.draw.circle(
            screen, circle_fill_color, self.rect.center, self.radius, width=0
        )
        pygame.draw.circle(
            screen, circle_color, self.rect.center, self.radius, width=10
        )
        text = self.font.render(f"{round(self.reward, 2)}", True, (44, 44, 44))
        screen.blit(text, (self.rect.center[0] - 20, self.rect.center[1] - 30))

    def click(self, pos):
        if not self.rect.collidepoint(pos):
            return None
        white = (255, 255, 255)
        self.details.fill(white)

        key_text = self.font.render(
            f"省份:{prov_map[self.key]}", True, (44, 44, 44), white
        )
        self.details.blit(key_text, (18, 82))

        s_str = [str(round(s, 2)) for s in self.storage]
        s_text = self.font.render(f"库存:{','.join(s_str)}", True, (44, 44, 44), white)
        self.details.blit(s_text, (18, 104))

        p_str = [str(p) for p in self.production]
        p_text = self.font.render(f"生产:{','.join(p_str)}", True, (35, 138, 32), white)
        self.details.blit(p_text, (18, 126))

        d_str = [str(d) for d in self.demand]
        d_text = self.font.render(f"消耗:{','.join(d_str)}", True, (251, 45, 45), white)
        self.details.blit(d_text, (18, 148))

        r_text = self.font.render(
            f"奖赏:{round(self.reward, 2)}", True, (12, 140, 210), white
        )
        self.details.blit(r_text, (18, 170))

        return self.details

    @staticmethod
    def _lighten_color(color, alpha=0.1):
        r = alpha * color[0] + (1 - alpha) * 255
        g = alpha * color[1] + (1 - alpha) * 255
        b = alpha * color[2] + (1 - alpha) * 255
        light_color = pygame.Color((r, g, b))
        return light_color


class Truck(object):
    def __init__(self, direction, start, end, trans_time, size=(20, 20)):
        self.dir = direction
        self.image = pygame.image.load(
            osp.join(resource_path, "img/truck.png")
        ).convert_alpha()
        self.image = pygame.transform.scale(self.image, size)
        self.rect = self.image.get_rect()
        self.rect.center = start
        self.font = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 14)

        self.init_pos = (self.rect.x, self.rect.y)
        self.total_frame = trans_time * FPS * SPD // 24
        self.update_frame = 0

        speed_x = 24 * (end[0] - start[0]) / (trans_time * FPS * SPD)
        speed_y = 24 * (end[1] - start[1]) / (trans_time * FPS * SPD)
        self.speed = (speed_x, speed_y)

        self.action = None
        self.details = pygame.Surface((200, 200))

    def update(self, action):
        self.action = action
        if self.update_frame < self.total_frame:
            self.update_frame += 1
            self.rect.x = self.init_pos[0] + self.speed[0] * self.update_frame
            self.rect.y = self.init_pos[1] + self.speed[1] * self.update_frame
        else:
            self.update_frame += 1
            if self.update_frame >= FPS * SPD:
                self.update_frame = 0
                self.rect.topleft = self.init_pos

    def draw(self, screen):
        total = sum(self.action)
        if total <= 0:  # 若货车运输量为0，则不显示
            return
        # 当货车在道路上时才显示
        if 0 < self.update_frame < self.total_frame:
            screen.blit(self.image, self.rect)
            # a_str = [str(round(a, 2)) for a in self.action]
            # a_text = self.font.render(f"{';'.join(a_str)}", True, (44, 44, 44), (238, 238, 238))
            # screen.blit(a_text, (self.rect.center[0] - 40, self.rect.center[1] - 30))

    def click(self, pos):
        if not self.rect.collidepoint(pos):
            return None
        white = (255, 255, 255)
        self.details.fill(white)

        dir_text = self.font.render(
            f"方向:{prov_map[self.dir[0]]}->{prov_map[self.dir[1]]}",
            True,
            (44, 44, 44),
            white,
        )
        self.details.blit(dir_text, (18, 148))

        a_str = [str(round(a, 2)) for a in self.action]
        a_text = self.font.render(f"运输:{','.join(a_str)}", True, (44, 44, 44), white)
        self.details.blit(a_text, (18, 170))

        return self.details
