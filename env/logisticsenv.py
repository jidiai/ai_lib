import json
import os.path
import sys
from pathlib import Path
from random import randint, sample
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, rgb2hex
from scipy.spatial import distance

import igraph
import pygame

from env.simulators.game import Game
from env.obs_interfaces.observation import DictObservation
import numpy as np
from utils.box import Box


current_dir = str(Path(__file__).resolve().parent)
resource_path = os.path.join(current_dir, "logistics", "resources")


def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config


class LogisticsEnv(Game, DictObservation):
    def __init__(self, conf):
        super().__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                         conf['game_name'], conf['agent_nums'], conf['obs_type'])
        map_path = os.path.join(current_dir, "logistics", "map_1.json")
        self.map_conf = load_config(map_path)
        # self.map_conf = generate_random_map()
        self.max_step = int(conf['max_step'])
        self.step_cnt = 0

        self.players = []
        # self.current_state = self.init_map(self.map_conf)
        # self.all_observes = self.get_all_observations()
        self.n_return = [0] * self.n_player
        self.won = {}
        self.init_info = None
        self.done = False
        self.interface_ctrl = None
        self.FPSClock = None
        self.joint_action_space = None
        self.info = {}
        self.render_mode = False
        self.render_start = False
        self.screen = None
        self.all_observes = self.reset()
        # 每个玩家的action space list, 可以根据player_id获取对应的single_action_space
        # self.joint_action_space = self.set_action_space()
        # self.info = {
        #     'upper_storages': [self.players[i].upper_storage for i in range(self.n_player)],
        #     'upper_capacity': [act[0].high.tolist() for act in self.joint_action_space]
        # }

    def init_map(self, conf):
        # 添加图中的节点
        vertices = conf['vertices'].copy()
        for vertex_info in vertices:
            key = vertex_info['key']
            self.add_vertex(key, vertex_info)

        # 添加图中的有向边
        edges = conf['edges'].copy()
        for edge_info in edges:
            start = edge_info['start']
            end = edge_info['end']
            self.add_edge(start, end, edge_info)

        # 对每个节点进行初始化
        init_state = []
        for i in range(self.n_player):
            self.players[i].update_init_storage()
            init_state.append(self.players[i].init_storage)

        return init_state

    def add_vertex(self, key, vertex_info):
        vertex = LogisticsVertex(key, vertex_info)
        self.players.append(vertex)

    def add_edge(self, start, end, edge_info):
        edge = LogisticsEdge(edge_info)
        start_vertex = self.players[start]
        start_vertex.add_neighbor(end, edge)

    def reset(self):
        self.step_cnt = 0
        self.players = []
        self.current_state = self.init_map(self.map_conf)
        self.all_observes = self.get_all_observations()
        self.n_return = [0] * self.n_player
        self.won = {}
        self.done = False

        self.joint_action_space = self.set_action_space()
        self.info = {
            'upper_storages': [self.players[i].upper_storage for i in range(self.n_player)],
            'upper_capacity': [act[0].high.tolist() for act in self.joint_action_space]
        }

        self.render_mode = False

        return self.all_observes

    def render_reset(self):
        network_data = self.get_network_data()
        self.FPSClock = pygame.time.Clock()
        self.interface_ctrl = LogisticsInterface(1000, 800, network_data, self.screen)

        self.info = {
            'upper_storages': [self.players[i].upper_storage for i in range(self.n_player)],
            'upper_capacity': [act[0].high.tolist() for act in self.joint_action_space],
            'reward': sum(self.n_return),
            'actual_actions': [[[0] * space[0].shape[0]] for space in self.joint_action_space]
        }
        self.info['actual_actions'] = self.bound_actions(self.info['actual_actions'])

        start_storages, productions, demands = [], [], []
        for i in range(self.n_player):
            start_storages.append(self.players[i].final_storage)
            productions.append(self.players[i].production)
            demands.append(self.players[i].demand)

        self.info.update({
            'start_storages': start_storages,
            'productions': productions,
            'demands': demands
        })

    def step(self, all_actions):
        self.step_cnt += 1

        all_actions = self.bound_actions(all_actions)
        self.info.update({'actual_actions': all_actions})

        self.current_state = self.get_next_state(all_actions)
        self.all_observes = self.get_all_observations()

        reward, single_rewards = self.get_reward(all_actions)
        done = self.is_terminal()
        self.info.update({
            'reward': reward,
            'single_rewards': single_rewards
        })

        return self.all_observes, single_rewards, done, "", self.info

    def bound_actions(self, all_actions):  # 对每个节点的动作进行约束
        bounded_actions = []

        for i in range(self.n_player):
            vertex = self.players[i]
            action = all_actions[i].copy()[0]
            actual_trans = sum(action)
            if vertex.init_storage < 0:  # 初始库存量为负（有货物缺口）
                bounded_actions.append([0] * len(action))
            elif actual_trans > vertex.init_storage:  # 运出的总货物量超过初始库存量
                # 每条运输途径的货物量进行等比例缩放
                bounded_action = [act * vertex.init_storage / actual_trans for act in action]
                bounded_actions.append(bounded_action)
            else:  # 合法动作
                bounded_actions.append(action)

        return bounded_actions

    def get_next_state(self, all_actions):
        assert len(all_actions) == self.n_player
        # 统计每个节点当天运出的货物量out_storages，以及接收的货物量in_storages
        out_storages, in_storages = [0] * self.n_player, [0] * self.n_player
        for i in range(self.n_player):
            action = all_actions[i]
            out_storages[i] = sum(action)
            connections = self.players[i].get_connections()
            for (act, nbr) in zip(action, connections):
                in_storages[nbr] += act

        # 更新每个节点当天的最终库存量以及下一天的初始库存量，
        # 并记录每个节点当天最开始的初始库存start_storages、生产量productions和消耗量demands，用于可视化
        next_state = []
        start_storages, productions, demands = [], [], []
        for i in range(self.n_player):
            start_storages.append(self.players[i].final_storage)
            productions.append(self.players[i].production)
            demands.append(self.players[i].demand)
            self.players[i].update_final_storage(out_storages[i], in_storages[i])
            self.players[i].update_init_storage()
            next_state.append(self.players[i].init_storage)
        self.info.update({
            'start_storages': start_storages,
            'productions': productions,
            'demands': demands
        })

        return next_state

    def get_dict_observation(self, current_state, player_id, info_before):
        obs = {
            "obs": current_state,
            "connected_player_index": self.players[player_id].get_connections(),
            "controlled_player_index": player_id
        }
        return obs

    def get_all_observations(self, info_before=''):
        all_obs = self.get_dict_many_observation(
            self.current_state,
            range(self.n_player),
            info_before
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
            high = []
            for j in vertex.get_connections():
                edge = vertex.get_edge(j)
                high.append(edge.upper_capacity)
            action_space_i = Box(np.zeros(len(high)), np.array(high), dtype=np.float64)
            action_space.append([action_space_i])

        return action_space

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_single_connections(self, player_id):
        return self.players[player_id].get_connections()

    def is_terminal(self):
        is_done = self.step_cnt >= self.max_step
        if is_done:
            self.done = True
        return is_done

    def get_network_data(self):
        network_data = {
            'n_vertex': self.n_player,
            'v_coords': self.map_conf.get('coords')
        }

        pd_gap, edges, edges_length = [], [], []
        for i in range(self.n_player):
            vertex = self.players[i]
            pd_gap.append(vertex.production - vertex.lambda_)
            for j in vertex.get_connections():
                edges.append((i, j))
                edges_length.append(vertex.get_edge(j).trans_time)
        network_data['pd_gap'] = pd_gap  # 记录每个节点生产量和平均消耗量之间的差距
        network_data['roads'] = edges
        network_data['roads_length'] = edges_length

        return network_data

    def get_render_data(self, current_state=None):
        render_data = {
            'day': self.step_cnt,
            'storages': self.info['start_storages'],
            'productions': self.info['productions'],
            'demands': self.info['demands'],
            'reward': self.info['reward']
        }

        actions = []
        for action_i in self.info['actual_actions']:
            if isinstance(action_i, np.ndarray):
                action_i = action_i.tolist()
            actions += action_i
        render_data['actions'] = actions

        return render_data

    def check_win(self):
        return '-1'

    def render(self):

        if not self.render_start:
            pygame.init()
            pygame.display.set_caption("Simple Logistics Simulator")
            self.screen = pygame.display.set_mode([1000, 800])
            self.render_start = True

        if not self.render_mode:
            self.render_reset()
            self.render_mode = True

        render_data = self.get_render_data()

        current_frame = 0
        while current_frame < FPS * SPD:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.interface_ctrl.refresh_background(render_data)
            self.interface_ctrl.move_trucks(render_data['actions'])
            current_frame += 1

            self.FPSClock.tick(FPS)
            pygame.display.update()


class LogisticsVertex(object):
    def __init__(self, key, info):
        self.key = key
        self.connectedTo = {}

        self.production = info['production']
        self.init_storage = 0
        self.final_storage = info['init_storage']
        self.upper_storage = info['upper_storage']
        self.store_cost = info['store_cost']
        self.loss_cost = info['loss_cost']
        self.storage_loss = 0  # 更新完当天的最终库存量后，统计当天的库存溢出量
        self.init_storage_loss = 0  # 因为每次状态更新会提前计算下一天的初始库存量，
        # 若不单独记录初始库存的溢出量，则会在计算每日reward时出错
        self.lambda_ = info['lambda']
        self.demand = 0

    def add_neighbor(self, nbr, edge):
        self.connectedTo.update({nbr: edge})

    def get_connections(self):
        return list(self.connectedTo.keys())

    def get_edge(self, nbr):
        return self.connectedTo.get(nbr)

    def get_demand(self):
        demand = np.random.poisson(lam=self.lambda_, size=1)
        return demand[0]

    def update_init_storage(self):
        self.demand = self.get_demand()
        self.init_storage = self.final_storage - self.demand + self.production
        self.init_storage_loss = 0
        if self.init_storage > self.upper_storage:  # 当天初始库存量超过存储上限
            self.init_storage_loss = self.init_storage - self.upper_storage
            self.init_storage = self.upper_storage

    def update_final_storage(self, out_storage, in_storage):
        self.final_storage = self.init_storage - out_storage + in_storage
        self.storage_loss = self.init_storage_loss
        if self.final_storage > self.upper_storage:  # 当天最终库存量超过存储上限
            self.storage_loss += (self.final_storage - self.upper_storage)
            self.final_storage = self.upper_storage

    def calc_reward(self, action, mu=30):
        connections = self.get_connections()
        assert len(action) == len(connections)
        # 舍弃超过库存货物造成的损失
        reward = -self.loss_cost * self.storage_loss

        # 当日运输货物的成本
        for (act, nbr) in zip(action, connections):
            edge = self.get_edge(nbr)
            reward -= (edge.trans_cost * edge.trans_time * act)

        if self.final_storage >= 0:  # 因库存盈余所导致的存储成本
            reward -= (self.store_cost * self.final_storage)
        else:  # 因库存空缺而加的惩罚项
            reward += (mu * self.final_storage)

        return reward


class LogisticsEdge(object):
    def __init__(self, info):
        self.upper_capacity = info['upper_capacity']
        self.trans_time = info['trans_time']
        self.trans_cost = info['trans_cost']


# NOTE: FPS*SPD应为24的倍数，否则可能导致货车到达终点时偏移仓库图标中心
FPS = 60  # Frame Per Second，帧率，即每秒播放的帧数
SPD = 4  # Second Per Day，游戏中每天所占的秒数


class Truck(object):
    def __init__(self, size, start_position, end_position, trans_time):
        self.image = pygame.image.load(os.path.join(resource_path, "img/truck.png")).convert_alpha()
        self.image = pygame.transform.scale(self.image, size)
        self.rect = self.image.get_rect()
        self.rect.center = start_position
        self.font = pygame.font.Font(os.path.join(resource_path, "font/arial.ttf"), 14)

        self.init_position = (self.rect.x, self.rect.y)
        self.total_frame = trans_time * FPS * SPD // 24
        self.update_frame = 0

        speed_x = 24 * (end_position[0] - start_position[0]) / (trans_time * FPS * SPD)
        speed_y = 24 * (end_position[1] - start_position[1]) / (trans_time * FPS * SPD)
        self.speed = (speed_x, speed_y)

    def update(self):
        if self.update_frame < self.total_frame:
            self.update_frame += 1
            self.rect.x = self.init_position[0] + self.speed[0] * self.update_frame
            self.rect.y = self.init_position[1] + self.speed[1] * self.update_frame
        else:
            self.update_frame += 1
            if self.update_frame >= FPS * SPD:
                self.update_frame = 0
                self.rect.topleft = self.init_position

    def draw(self, screen, action):
        if action <= 0:  # 若货车运输量为0，则不显示
            return
        # 当货车在道路上时才显示
        if 0 < self.update_frame < self.total_frame:
            screen.blit(self.image, self.rect)
            text = self.font.render(f"{round(action, 2)}", True, (0, 0, 0), (255, 255, 255))
            text_rect = text.get_rect()
            text_rect.centerx = self.rect.centerx
            text_rect.y = self.rect.y - 12
            screen.blit(text, text_rect)


class LogisticsInterface(object):
    def __init__(self, width, height, network_data, screen):
        self.width = width
        self.height = height
        self.v_radius = 36
        self.n_vertex = network_data['n_vertex']
        self.pd_gap = network_data['pd_gap']  # 每个节点生产量和平均消耗量之间的差距
        self.roads = network_data['roads']
        self.roads_length = network_data['roads_length']
        self.v_coords = self._spread_vertex(network_data['v_coords'])
        self.v_colors = []

        self.screen = screen
        # self.screen = pygame.display.set_mode([width, height])
        self.screen.fill("white")

        self.font1 = pygame.font.Font(os.path.join(resource_path, "font/arial.ttf"), 24)
        self.font2 = pygame.font.Font(os.path.join(resource_path, "font/arial.ttf"), 18)
        self.font3 = pygame.font.Font(os.path.join(resource_path, "font/arial.ttf"), 14)
        self.font4 = pygame.font.Font(os.path.join(resource_path, "font/simhei.ttf"), 14)
        self.p_img = pygame.image.load(os.path.join(resource_path, "img/produce.png")).convert_alpha()
        self.p_img = pygame.transform.scale(self.p_img, (16, 16))
        self.d_img = pygame.image.load(os.path.join(resource_path, "img/demand.png")).convert_alpha()
        self.d_img = pygame.transform.scale(self.d_img, (14, 14))

        self.background = self.init_background()
        self.trucks = self.init_trucks()

    def init_background(self):
        # 绘制道路
        for (v_start, v_end) in self.roads:
            start = self.v_coords[v_start]
            end = self.v_coords[v_end]
            self._rotated_road(start, end, height=12,
                               border_color=(252, 122, 90), fill_color=(255, 172, 77))

        # 绘制仓库节点
        norm = Normalize(vmin=min(self.pd_gap) - 2,
                         vmax=max(self.pd_gap) + 2)  # 数值映射范围（略微扩大）
        color_map = get_cmap('RdYlGn')  # 颜色映射表
        for coord, gap in zip(self.v_coords, self.pd_gap):
            rgb = color_map(norm(gap))[:3]
            color = pygame.Color(rgb2hex(rgb))
            light_color = self._lighten_color(color)
            pygame.draw.circle(self.screen, light_color, coord, self.v_radius, width=0)
            pygame.draw.circle(self.screen, color, coord, self.v_radius, width=2)
            self.v_colors.append(light_color)

        # 加入固定的提示
        self.add_notation()

        # 保存当前初始化的背景，便于后续刷新时使用
        background = self.screen.copy()
        return background

    @staticmethod
    def _lighten_color(color, alpha=0.1):
        r = alpha * color.r + (1 - alpha) * 255
        g = alpha * color.g + (1 - alpha) * 255
        b = alpha * color.b + (1 - alpha) * 255
        light_color = pygame.Color((r, g, b))
        return light_color

    def _spread_vertex(self, v_coords):
        if not v_coords:  # 若没有指定相对坐标，则随机将节点分布到画布上
            g = igraph.Graph()
            for i in range(self.n_vertex):
                g.add_vertex(i)
            g.add_edges(self.roads)
            layout = g.layout_kamada_kawai()
            layout_coords = np.array(layout.coords).T
        else:  # 否则使用地图数据中指定的节点相对坐标
            layout_coords = np.array(v_coords).T

            # 将layout的坐标原点对齐到左上角
        layout_coords[0] = layout_coords[0] - layout_coords[0].min()
        layout_coords[1] = layout_coords[1] - layout_coords[1].min()

        # 将layout的坐标映射到画布坐标，并将图形整体居中
        stretch_rate = min((self.width - 2 * self.v_radius - 20) / layout_coords[0].max(),
                           (self.height - 2 * self.v_radius - 20) / layout_coords[1].max())
        margin_x = (self.width - layout_coords[0].max() * stretch_rate) // 2
        margin_y = (self.height - layout_coords[1].max() * stretch_rate) // 2
        vertex_coord = []
        for i in range(self.n_vertex):
            x = margin_x + int(layout_coords[0, i] * stretch_rate)
            y = margin_y + int(layout_coords[1, i] * stretch_rate)
            vertex_coord.append((x, y))

        return vertex_coord

    def _rotated_road(self, start, end, height, border_color=(0, 0, 0), fill_color=None):
        length = distance.euclidean(start, end)
        sin = (end[1] - start[1]) / length
        cos = (end[0] - start[0]) / length

        vertex = lambda e1, e2: (
            start[0] + (e1 * length * cos + e2 * height * sin) / 2,
            start[1] + (e1 * length * sin - e2 * height * cos) / 2
        )
        vertices = [vertex(*e) for e in [(0, -1), (0, 1), (2, 1), (2, -1)]]

        if not fill_color:
            pygame.draw.polygon(self.screen, border_color, vertices, width=3)
        else:
            pygame.draw.polygon(self.screen, fill_color, vertices, width=0)
            pygame.draw.polygon(self.screen, border_color, vertices, width=2)

    def init_trucks(self):
        truck_list = []
        for i, (v_start, v_end) in enumerate(self.roads):
            start = self.v_coords[v_start]
            end = self.v_coords[v_end]
            road_length = self.roads_length[i]
            truck = Truck((32, 32), start, end, road_length)
            truck_list.append(truck)

        return truck_list

    def move_trucks(self, actions):
        for truck, action in zip(self.trucks, actions):
            truck.update()
            truck.draw(self.screen, action)

    def refresh_background(self, render_data):
        day = render_data['day']
        storages = render_data['storages']
        productions = render_data['productions']
        demands = render_data['demands']
        reward = render_data['reward']

        self.screen.blit(self.background, (0, 0))
        day_text = self.font1.render(f"Day {day}", True, (44, 44, 44), (255, 255, 255))
        self.screen.blit(day_text, (10, 5))
        r_text = self.font2.render(f"Reward: {round(reward, 2)}", True, (44, 44, 44), (255, 255, 255))
        self.screen.blit(r_text, (10, 35))

        for coord, s, p, d, color in zip(self.v_coords, storages, productions, demands, self.v_colors):
            s_text = self.font3.render(f"{round(s, 2)}", True, (44, 44, 44), color)
            s_text_rect = s_text.get_rect()
            s_text_rect.centerx, s_text_rect.y = coord[0], coord[1] - 24
            self.screen.blit(s_text, s_text_rect)

            p_text = self.font3.render(f"+{round(p, 2)}", True, (35, 138, 32), color)
            p_text_rect = p_text.get_rect()
            p_text_rect.centerx, p_text_rect.y = coord[0] + 8, coord[1] - 7
            self.screen.blit(p_text, p_text_rect)
            p_img_rect = self.p_img.get_rect()
            p_img_rect.centerx, p_img_rect.y = coord[0] - 14, coord[1] - 7
            self.screen.blit(self.p_img, p_img_rect)

            d_text = self.font3.render(f"-{round(d, 2)}", True, (251, 45, 45), color)
            d_text_rect = d_text.get_rect()
            d_text_rect.centerx, d_text_rect.y = coord[0] + 8, coord[1] + 10
            self.screen.blit(d_text, d_text_rect)
            d_img_rect = self.d_img.get_rect()
            d_img_rect.centerx, d_img_rect.y = coord[0] - 14, coord[1] + 10
            self.screen.blit(self.d_img, d_img_rect)

    def add_notation(self):
        text1 = self.font4.render("黑:初始库存量", True, (44, 44, 44), (255, 255, 255))
        self.screen.blit(text1, (10, 60))

        text2 = self.font4.render(":生产量", True, (35, 138, 32), (255, 255, 255))
        self.screen.blit(text2, (24, 80))
        self.screen.blit(self.p_img, (9, 80))

        text3 = self.font4.render(":消耗量", True, (251, 45, 45), (255, 255, 255))
        self.screen.blit(text3, (24, 100))
        self.screen.blit(self.d_img, (9, 100))


MIN_PRODUCTION = 10
MAX_PRODUCTION = 50

MIN_INIT_STORAGE = 10
MAX_INIT_STORAGE = 120

MIN_UPPER_STORAGE = 30
MAX_UPPER_STORAGE = 150

MIN_STORE_COST = 1
MAX_STORE_COST = 5

MIN_LOSS_COST = 1
MAX_LOSS_COST = 5

MIN_LAMBDA = 10
MAX_LAMBDA = 50

MIN_UPPER_CAPACITY = 8
MAX_UPPER_CAPACITY = 20

MIN_TRANS_TIME = 4
MAX_TRANS_TIME = 24

MIN_TRANS_COST = 1
MAX_TRANS_COST = 3


def generate_random_map():
    num_vertex = 10

    vertices, edges, connections = [], [], []
    for v in range(num_vertex):
        vertex = {
            "key": v,
            "production": randint(MIN_PRODUCTION, MAX_PRODUCTION),
            "init_storage": randint(MIN_INIT_STORAGE, MAX_INIT_STORAGE),
            "upper_storage": randint(MIN_UPPER_STORAGE, MAX_UPPER_STORAGE),
            "store_cost": randint(MIN_STORE_COST, MAX_STORE_COST) / 10,
            "loss_cost": randint(MIN_LOSS_COST, MAX_LOSS_COST) / 10,
            "lambda": randint(MIN_LAMBDA, MAX_LAMBDA)
        }
        vertices.append(vertex)

    num_circle = randint(3, num_vertex)
    used_vertex = sample(list(range(num_vertex)), num_circle)
    for i in range(num_circle):
        edge = {
            "start": used_vertex[i],
            "end": used_vertex[(i + 1) % num_circle],
            "upper_capacity": randint(MIN_UPPER_CAPACITY, MAX_UPPER_CAPACITY),
            "trans_time": randint(MIN_TRANS_TIME, MAX_TRANS_TIME),
            "trans_cost": randint(MIN_TRANS_COST, MAX_TRANS_COST) / 10
        }
        edges.append(edge)

    for v in range(num_vertex):
        if v in used_vertex:
            continue

        in_num = randint(1, len(used_vertex) - 1)
        in_vertex = sample(used_vertex, in_num)
        for i in in_vertex:
            edge = {
                "start": i,
                "end": v,
                "upper_capacity": randint(MIN_UPPER_CAPACITY, MAX_UPPER_CAPACITY),
                "trans_time": randint(MIN_TRANS_TIME, MAX_TRANS_TIME),
                "trans_cost": randint(MIN_TRANS_COST, MAX_TRANS_COST) / 10
            }
            edges.append(edge)

        left_vertex = list(set(used_vertex).difference(set(in_vertex)))
        out_num = randint(1, len(used_vertex) - in_num)
        out_vertex = sample(left_vertex, out_num)
        for i in out_vertex:
            edge = {
                "start": v,
                "end": i,
                "upper_capacity": randint(MIN_UPPER_CAPACITY, MAX_UPPER_CAPACITY),
                "trans_time": randint(MIN_TRANS_TIME, MAX_TRANS_TIME),
                "trans_cost": randint(MIN_TRANS_COST, MAX_TRANS_COST) / 10
            }
            edges.append(edge)

        used_vertex.append(v)

    map_data = {
        "n_vertex": num_vertex,
        "vertices": vertices,
        "edges": edges
    }

    return map_data
