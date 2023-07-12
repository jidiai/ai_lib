# -*- coding:utf-8  -*-
# Time  : 2022/2/11 下午3:18
# Author: Yahui Cui
import copy
import os
import random
import numpy as np

import pygame
from PIL import Image, ImageDraw

from .simulators.game import Game

from utils.discrete import Discrete


class DeliveryGame(Game):

    def __init__(self, conf, seed=None):
        super(DeliveryGame, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                           conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.seed = seed
        self.set_seed()
        self.board_height = N
        self.board_width = N
        self.env_core = Delivery(self.n_player)
        self.joint_action_space = self.set_action_space()
        self.current_state = self.env_core.reset()
        self.init_info = self.env_core.get_init_info()
        self.all_observes = self.get_all_observations()
        self.step_cnt = 0
        self.n_return = [0] * self.n_player
        self.done = False
        self.won = {}

    def reset(self):
        self.current_state = self.env_core.reset()
        self.init_info = self.env_core.get_init_info()
        self.all_observes = self.get_all_observations()
        self.step_cnt = 0
        self.n_return = [0] * self.n_player
        self.done = False
        self.won = {}

        return self.all_observes

    def step(self, joint_action):
        self.is_valid_action(joint_action)
        info_before = ''
        joint_action_decode = self.decode(joint_action)
        self.current_state, step_rewards, self.done, info_after = self.env_core.step(joint_action_decode)
        self.all_observes = self.get_all_observations()
        done = self.is_terminal()
        self.set_n_return(step_rewards)
        self.step_cnt += 1
        return self.all_observes, step_rewards, done, info_before, info_after

    def decode(self, joint_action):
        # If action[0] doesn't contain 1, the agent will not move.
        joint_action_decode = [[action[0].index(1) if 1 in action[0] else 4, action[1], action[2], action[3]]
                               for action in joint_action]

        return joint_action_decode

    def is_valid_action(self, joint_action):
        if len(joint_action) != self.n_player:
            raise Exception("Input joint action dimension should be {}, not {}.".format(
                self.n_player, len(joint_action)))

        for i in range(self.n_player):
            if len(joint_action[i]) != 4:
                raise Exception("The input action should have 4 parts, not {}".format(len(joint_action[i])))
            if len(joint_action[i][0]) != self.joint_action_space[i][0].n:
                raise Exception("The first action dimension for player {} should be {}, not {}.".format(
                    i, self.joint_action_space[i][0].n, len(joint_action[i][0])))
            if len(joint_action[i][1]) != self.joint_action_space[i][1].n:
                raise Exception("The second action dimension for player {} should be {}, not {}.".format(
                    i, self.joint_action_space[i][1].n, len(joint_action[i][1])))
            if len(joint_action[i][2]) != self.joint_action_space[i][2].n:
                raise Exception("The third action dimension for player {} should be {}, not {}.".format(
                    i, self.joint_action_space[i][2].n, len(joint_action[i][2])))
            if len(joint_action[i][3]) != self.joint_action_space[i][3].n:
                raise Exception("The fourth action dimension for player {} should be {}, not {}.".format(
                    i, self.joint_action_space[i][3].n, len(joint_action[i][3])))

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_all_observations(self):
        all_observes = []
        for i in range(self.n_player):
            each_obs = copy.deepcopy(self.current_state)
            each = {"obs": each_obs, "controlled_player_index": i}
            all_observes.append(each)
        return all_observes

    def set_action_space(self):
        action_spaces = [[Discrete(5), Discrete(DISTRIBUTE_NUM), Discrete(CAPACITY_RIDER), Discrete(CAPACITY_DISTRIBUTE)]
                         for _ in range(self.n_player)]
        return action_spaces

    def is_terminal(self):
        return self.done

    def set_n_return(self, step_rewards):
        for i in range(self.n_player):
            self.n_return[i] += step_rewards[i]

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

    def all_equals(self, list_to_compare):
        return len(set(list_to_compare)) == 1

    def set_seed(self, seed=None):
        if not seed:
            seed = self.seed
        random.seed(seed)
        if not self.seed:
            self.seed = seed


# [up, down, left, right, not move]
DIRECTION = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]]
N = 16
UPDATE_FREQUENCY = 10
CAPACITY_RIDER = 5
CAPACITY_GERNATION = 20
CAPACITY_DISTRIBUTE = 10
RESTAURANT_NUM = 10
CUSTOMER_NUM = 20
GRASS_NUM = 10
TOTAL_STEP = 500
DISTRIBUTE_DISTANCE = 5
DISTRIBUTE_NUM = 20
PICK_UP_TIMES = 2
TIME_INTERVAL_END_TIME = 400

GRID_UNIT = 40
FIX = 8

# 0: obstacles 1: roads horizontal 2: available positions for customers and restaurants
# 3: customers and restaurants(modified after init) 4: roads vertical 5: roads intersection 6: grass
# 0, 2, 6: can't be reached by riders; 1 3 4 5: can be reached by riders;
# only roads, restaurants and customers can be reached by riders
MAP = [[0, 0, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 0, 0],
       [0, 0, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 0, 0],
       [2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2],
       [1, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 1],
       [2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2],
       [2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2],
       [1, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 1],
       [2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2],
       [2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2],
       [1, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 1],
       [2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2],
       [2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2],
       [1, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 1],
       [2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2],
       [0, 0, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 0, 0],
       [0, 0, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 0, 0]]

MAP2 = [[0, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 0],
        [0, 2, 5, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 2, 0],
        [0, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2],
        [1, 1, 5, 2, 5, 1, 1, 1, 1, 5, 2, 0, 2, 5, 1, 1],
        [2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 0, 2, 4, 2, 2],
        [0, 2, 4, 1, 5, 1, 5, 2, 2, 4, 2, 0, 2, 4, 2, 0],
        [2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 0, 2, 4, 2, 2],
        [1, 1, 5, 2, 0, 2, 4, 2, 2, 5, 1, 1, 1, 5, 1, 1],
        [2, 2, 2, 0, 0, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 0, 0, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 5, 2, 2, 4, 2, 0, 2, 5, 1, 1],
        [2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 0, 2, 4, 2, 2],
        [2, 2, 2, 0, 0, 2, 4, 2, 2, 4, 2, 0, 2, 4, 2, 2],
        [1, 1, 5, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 5, 1, 1],
        [2, 2, 5, 1, 1, 1, 5, 1, 5, 5, 1, 1, 1, 5, 2, 2],
        [0, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 0]]


class Delivery:
    def __init__(self, agent_num):
        self.agent_num = agent_num
        self.agents = []
        self.customers = {}
        self.restaurants = {}
        self.pos2restaurant = {}
        self.total_order = 0
        self.step_cnt = 0
        self.pygame_render = False
        # 0: obstacle; 1: road
        self.board = [[0 for _ in range(N)] for _ in range(N)]
        self.road = []
        self.edges = []
        self.horizontal = []
        self.vertical = []
        self.intersection = []
        self.grass = []
        self.distribute_map = {}
        self.new_generated_orders = []
        self.set_up()
        self.step_rewards = [0.0 for _ in range(len(self.agents))]
        self.total_rewards = [0.0 for _ in range(len(self.agents))]
        self.orders_sent_step = [[] for _ in range(len(self.agents))]
        self.orders_thrown_step = [[] for _ in range(len(self.agents))]
        self.orders_failed_pick_step = [[] for _ in range(len(self.agents))]
        self.orders_out_time_step = [[] for _ in range(len(self.agents))]

    def reset(self):
        self.agents = []
        self.customers = {}
        self.restaurants = {}
        self.pos2restaurant = {}
        self.total_order = 0
        self.step_cnt = 0
        self.pygame_render = False

        self.board = [[0 for _ in range(N)] for _ in range(N)]
        self.road = []
        self.edges = []
        self.horizontal = []
        self.vertical = []
        self.intersection = []
        self.grass = []
        self.distribute_map = {}
        self.new_generated_orders = []
        self.set_up()
        self.step_rewards = [0.0 for _ in range(len(self.agents))]
        self.total_rewards = [0.0 for _ in range(len(self.agents))]
        self.orders_sent_step = [[] for _ in range(len(self.agents))]
        self.orders_thrown_step = [[] for _ in range(len(self.agents))]
        self.orders_failed_pick_step = [[] for _ in range(len(self.agents))]
        self.orders_out_time_step = [[] for _ in range(len(self.agents))]

        current_state = self.get_current_state()
        return current_state

    def set_up(self):
        # add road
        # all_pos_key: add available positions for customers and restaurants
        all_pos_keys = []
        for row in range(N):
            for col in range(N):
                if MAP2[row][col] == 1 or MAP2[row][col] == 4 or MAP2[row][col] == 5:
                    self.board[row][col] = MAP2[row][col]
                    key = pos2key([row, col])
                    self.road.append(key)
                    if MAP2[row][col] == 1:
                        self.horizontal.append(key)
                    elif MAP2[row][col] == 4:
                        self.vertical.append(key)
                    elif MAP2[row][col] == 5:
                        self.intersection.append(key)
                elif MAP2[row][col] == 2:
                    self.board[row][col] = 2
                    key = pos2key([row, col])
                    all_pos_keys.append(key)
                    self.edges.append(key)

        # generate agents
        for agent_id in range(self.agent_num):
            # init_pos_key = random.randint(0, N * N - 1)
            init_pos_key = random.choice(self.road)
            init_pos = key2pos(init_pos_key)
            new_agent = Agent(agent_id, init_pos)
            self.agents.append(new_agent)

        # generate restaurants
        for restaurant_id in range(RESTAURANT_NUM):
            init_pos_key = random.choice(all_pos_keys)
            all_pos_keys.remove(init_pos_key)
            init_pos = key2pos(init_pos_key)
            new_restaurant = Restaurant(restaurant_id, init_pos)
            self.restaurants[restaurant_id] = new_restaurant
            self.pos2restaurant[init_pos_key] = restaurant_id
            self.board[init_pos[0]][init_pos[1]] = 3

        # generate customers
        for customer_id in range(CUSTOMER_NUM):
            init_pos_key = random.choice(all_pos_keys)
            all_pos_keys.remove(init_pos_key)
            init_pos = key2pos(init_pos_key)
            new_customer = Customer(customer_id, init_pos)
            self.customers[customer_id] = new_customer
            self.board[init_pos[0]][init_pos[1]] = 3

        # initial orders
        self.generate_orders(self.step_cnt)
        self.distribute()

        # generate grass
        all_grass = []
        for row in range(N):
            for col in range(N):
                if self.board[row][col] == 0 or self.board[row][col] == 2:
                    key = pos2key([row, col])
                    all_grass.append(key)

        for _ in range(GRASS_NUM):
            init_pos_key = random.choice(all_grass)
            all_grass.remove(init_pos_key)
            init_pos = key2pos(init_pos_key)
            self.board[init_pos[0]][init_pos[1]] = 6
            self.grass.append(init_pos_key)

    def step(self, actions):

        self.step_rewards = [0.0 for _ in range(len(self.agents))]

        # for info_after
        self.orders_sent_step = [[] for _ in range(len(self.agents))]
        self.orders_thrown_step = [[] for _ in range(len(self.agents))]
        self.orders_failed_pick_step = [[] for _ in range(len(self.agents))]
        self.orders_out_time_step = [[] for _ in range(len(self.agents))]

        move_actions = [action[0] for action in actions]
        pick_order_actions = [action[1] for action in actions]
        put_order_actions = [action[2] for action in actions]
        grab_order_actions = [action[3] for action in actions]

        # update agent position
        for i in range(len(self.agents)):
            agent = self.agents[i]
            agent.update_position(move_actions[i], self.board)

        # check put_order_actions
        for i in range(len(self.agents)):
            agent = self.agents[i]
            put_order_action = put_order_actions[i]
            del_index = []
            for j in range(len(agent.order_list)):
                cur_order = agent.order_list[j]
                if put_order_action[j] == 1:
                    del_index.append(j)
                    if check_position(agent.position, self.customers[cur_order.customer_id].position) \
                            and cur_order.end_time >= self.step_cnt:
                        self.step_rewards[i] += cur_order.distance  # calculate_distance(self.customers[cur_order.customer_id].position,
                                                                    #   self.restaurants[cur_order.restaurant_id].position)
                        self.orders_sent_step[agent.agent_id].append(cur_order)
                    else:
                        self.step_rewards[i] -= cur_order.distance/2        #penalty when order overtime
                        self.orders_thrown_step[agent.agent_id].append(cur_order)
                elif cur_order.end_time < self.step_cnt:
                    del_index.append(j)
                    self.step_rewards[i] -= cur_order.distance/2
                    self.orders_out_time_step[agent.agent_id].append(cur_order)
            tmp = [i for num, i in enumerate(agent.order_list) if num not in del_index]
            agent.order_list = tmp

        # check orders_to_pick
        # move order from order_to_pick to order_list, remove order from restaurant
        for i in range(len(self.agents)):
            agent = self.agents[i]
            pick_order_action = pick_order_actions[i]
            del_index = []
            for j, order_to_pick in enumerate(agent.orders_to_pick):
                # if the order is out of end_time, drop the order
                if order_to_pick.end_time < self.step_cnt:
                    del_index.append(j)
                    self.orders_out_time_step[agent.agent_id].append(order_to_pick)
                    order_to_pick.remove = True
                elif pick_order_action[j] == 1:
                    if check_position(agent.position, self.restaurants[order_to_pick.restaurant_id].position) \
                            and len(agent.order_list) < CAPACITY_RIDER:
                        agent.order_list.append(order_to_pick)
                        order_to_pick.remove = True
                        del_index.append(j)
            tmp = [i for num, i in enumerate(agent.orders_to_pick) if num not in del_index]
            agent.orders_to_pick = tmp

        for restaurant_id, restaurant in self.restaurants.items():
            del_index = []
            for j, order in enumerate(restaurant.order_list):
                if order.end_time < self.step_cnt:
                    assert order.distributed == -1 or order.distributed==1
                    if order.distributed == 1:
                        self.step_rewards[order.rider_id] -= order.distance/2       #penalty when order is picked but overtime
                    del_index.append(j)
                elif order.remove:
                    del_index.append(j)
            tmp = [i for num, i in enumerate(restaurant.order_list) if num not in del_index]
            restaurant.order_list = tmp

        # generate new orders for restaurants
        if (self.step_cnt + 1) % UPDATE_FREQUENCY == 0:
            self.generate_orders(self.step_cnt)

        self.accept(grab_order_actions)

        self.distribute()

        current_state = self.get_current_state()

        self.step_cnt += 1
        done = self.step_cnt >= TOTAL_STEP

        if done:
            # when done, if there are still orders to be picked up or ordered hasn't been sent
            for i in range(len(self.agents)):
                agent = self.agents[i]
                # self.step_rewards[i] -= 100 * len(agent.orders_to_pick)
                # self.step_rewards[i] -= 100 * len(agent.order_list)
                self.orders_failed_pick_step[agent.agent_id] += agent.orders_to_pick
                self.orders_thrown_step[agent.agent_id] += agent.order_list

        for i in range(len(self.agents)):
            self.total_rewards[i] += self.step_rewards[i]

        info = self.get_info_after()

        return current_state, self.step_rewards, done, info

    def generate_orders(self, step_cnt):
        tmp_generation = []
        while len(tmp_generation) < CAPACITY_GERNATION:
            restaurant_id = random.randint(0, len(self.restaurants) - 1)
            restaurant = self.restaurants[restaurant_id]
            order_id = self.total_order
            customer_id = random.randint(0, len(self.customers) - 1)
            start_time = step_cnt
            distance = calculate_distance(restaurant.position, self.customers[customer_id].position)
            end_time = random.randint(step_cnt + distance, step_cnt + TIME_INTERVAL_END_TIME)
            new_order = Order(order_id, customer_id, restaurant_id, start_time, end_time, -1, False, distance)
            restaurant.order_list.append(new_order)
            tmp_generation.append(new_order)
            self.total_order += 1

    def accept(self, grab_order_actions):
        # distribute orders in restaurant to riders whose distance < DISTRIBUTE_DISTANCE.
        # Each rider can be distributed up to 20 orders.
        self.distribute_map = {}
        new_order2agents = {}
        for agent in self.agents:
            self.distribute_map[agent.agent_id] = []
            grab_action = grab_order_actions[agent.agent_id]

            for j, new_order in enumerate(self.new_generated_orders):
                if new_order.order_id not in new_order2agents.keys():
                    new_order2agents[new_order.order_id] = []
                if grab_action[j] == 1:
                    to_order_dist = calculate_distance(agent.position,
                                                       self.restaurants[new_order.restaurant_id].position)
                    new_order2agents[new_order.order_id].append((agent, to_order_dist))

        for _, new_order in enumerate(self.new_generated_orders):

            selected_agents = new_order2agents[new_order.order_id]
            selected_agents_sort = sorted(selected_agents, key=lambda pair: pair[1])

            if len(selected_agents_sort) == 0:
                continue

            queue = [[selected_agents_sort[0]]]
            degree = 0
            for i in range(1, len(selected_agents_sort)):
                if selected_agents_sort[i][1] == queue[degree][-1][1]:
                    queue[degree].append(selected_agents_sort[i])
                else:
                    queue.append([])
                    degree += 1
                    queue[degree].append(selected_agents_sort[i])
            while queue:
                next_agents = queue[0]
                queue.pop(0)
                while next_agents:
                    idx = random.randint(0, len(next_agents) - 1)
                    agent_idx = next_agents[idx][0].agent_id
                    next_agents.pop(idx)
                    if len(self.agents[agent_idx].orders_to_pick) < DISTRIBUTE_NUM:
                        self.agents[agent_idx].orders_to_pick.append(new_order)
                        new_order.distributed = 1
                        new_order.rider_id = agent_idx
                        self.distribute_map[agent_idx].append(new_order)
                        break
                if new_order.distributed == 1:
                    break

    def distribute(self):
        tmp_not_distributed = []
        self.new_generated_orders = []
        total_order = 0
        for rest_id, rest in self.restaurants.items():
            for order in rest.order_list:
                total_order += 1
                if order.distributed == -1 and order.end_time > self.step_cnt:
                    tmp_not_distributed.append(order)

        if len(tmp_not_distributed) < CAPACITY_DISTRIBUTE:
            self.new_generated_orders = tmp_not_distributed
        else:
            self.new_generated_orders = random.sample(tmp_not_distributed, CAPACITY_DISTRIBUTE)

    def get_current_state(self):
        return {
            "agents": [agent2dict(agent) for agent in self.agents],
            "restaurants": [restaurant2dict(restaurant) for _, restaurant in self.restaurants.items()],
            "customers": [customer2dict(customer) for _, customer in self.customers.items()],
            "distributed_orders": copy.deepcopy([(agent_id, [order2dict(order) for order in agent_orders])
                                                 for agent_id, agent_orders in self.distribute_map.items()]),
            "roads": copy.deepcopy([[key2pos(pos)[0], key2pos(pos)[1]] for pos in self.road]),
            "new_orders": [order2dict(order) for order in self.new_generated_orders],
            "current_step": self.step_cnt}

    def get_info_after(self):
        return {
            "agents": [agent2dict_info_after(agent) for agent in self.agents],
            "restaurants": [restaurant2dict_info_after(restaurant) for _, restaurant in self.restaurants.items()],
            # "customers": [customer2dict(customer) for _, customer in self.customers.items()],
            "distributed_orders": copy.deepcopy([(agent_id, [order.order_id for order in agent_orders])
                                                 for agent_id, agent_orders in self.distribute_map.items()]),
            "orders_sent_step": copy.deepcopy([(agent_id, [order.order_id for order in self.orders_sent_step[agent_id]])
                                               for agent_id in range(len(self.agents))]),
            "orders_thrown_step": copy.deepcopy([(agent_id, [order.order_id for order in
                                                             self.orders_thrown_step[agent_id]])
                                                 for agent_id in range(len(self.agents))]),
            "orders_failed_pick_step": copy.deepcopy([(agent_id, [order.order_id for order in
                                                                  self.orders_failed_pick_step[agent_id]])
                                                      for agent_id in range(len(self.agents))]),
            "orders_out_time_step": copy.deepcopy([(agent_id, [order.order_id for order in
                                                               self.orders_out_time_step[agent_id]])
                                                   for agent_id in range(len(self.agents))]),
            "total_rewards": copy.deepcopy(self.total_rewards)
        }

    def get_init_info(self):
        return {
            "agents": [agent2dict_info_after(agent) for agent in self.agents],
            "restaurants": [restaurant2dict_info_after(restaurant) for _, restaurant in self.restaurants.items()],
            "customers": [customer2dict(customer) for _, customer in self.customers.items()],
            "roads_vertical": copy.deepcopy([[key2pos(pos)[0], key2pos(pos)[1]] for pos in self.vertical]),
            "roads_horizontal": copy.deepcopy([[key2pos(pos)[0], key2pos(pos)[1]] for pos in self.horizontal]),
            "roads_intersection": copy.deepcopy([[key2pos(pos)[0], key2pos(pos)[1]] for pos in self.intersection]),
            "grass": copy.deepcopy([[key2pos(pos)[0], key2pos(pos)[1]] for pos in self.grass]),
            "distributed_orders": copy.deepcopy([(agent_id, [order.order_id for order in agent_orders])
                                                 for agent_id, agent_orders in self.distribute_map.items()])
        }

    def render(self, fps=1):
        if not self.pygame_render:
            # images
            resource_path_rider = os.path.join(os.path.dirname(__file__), "images", "delivery", "rider.png")
            resource_path_restaurant = os.path.join(os.path.dirname(__file__), "images", "delivery", "restaurant.png")
            resource_path_customer = os.path.join(os.path.dirname(__file__), "images", "delivery", "customer.png")
            self.images = {
                "agents": [Bitmap(Image.open(resource_path_rider), GRID_UNIT, (0, 191, 255))
                           for _ in range(len(self.agents))],
                "restaurants": [Bitmap(change_background(Image.open(resource_path_restaurant), (255, 255, 255)),
                                       GRID_UNIT - 1, (0, 0, 0)) for _ in range(len(self.restaurants))],
                "customers": [
                    Bitmap(change_background(Image.open(resource_path_customer), (255, 255, 255)), GRID_UNIT - 1,
                           (0, 0, 0)) for _ in range(len(self.customers))]
            }
            pygame.init()
            self.grid = Delivery.init_board(N, N, GRID_UNIT, color=(255,255,255))
            self.screen = pygame.display.set_mode(self.grid.size)
            # pygame.display.set_caption(self.game_name)
            self.game_tape = []

            self.clock = pygame.time.Clock()
            self.pygame_render = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        pygame.surfarray.blit_array(self.screen, self.render_board().transpose(1, 0, 2))
        debug(f'Step {self.step_cnt}', c='black')
        pygame.display.flip()

        self.clock.tick(fps)

    @staticmethod
    def init_board(width, height, grid_unit, color=(148, 0, 211)):
        im = Image.new(mode="RGB", size=(width * grid_unit, height * grid_unit), color=color)
        draw = ImageDraw.Draw(im)
        for x in range(0, width):
            draw.line(((x * grid_unit, 0), (x * grid_unit, height * grid_unit)), fill=(0, 0, 0))
        for y in range(0, height):
            draw.line(((0, y * grid_unit), (width * grid_unit, y * grid_unit)), fill=(0, 0, 0))
        return im

    def render_board(self):
        extra_info = {}
        for i in range(len(self.agents)):
            agent = self.agents[i]
            x = agent.position[0]
            y = agent.position[1]
            self.images["agents"][i].set_pos(x, y)
            # print("position in render {}, {}".format(pos, i))
            if (x, y) not in extra_info.keys():
                extra_info[(x, y)] = 'A_' + str(i) + " #" + str(len(agent.order_list))
            else:
                extra_info[(x, y)] += '\n' + 'A_' + str(i) + " #" + str(len(agent.order_list))

            if abs(self.step_rewards[i] - 0.0) > 0.0000001:
                extra_info[(x, y)] += '\n' + 'Reward' + '\n' + str(self.step_rewards[i])

        for restaurant_id, restaurant in self.restaurants.items():
            x = restaurant.position[0]
            y = restaurant.position[1]
            self.images["restaurants"][restaurant_id].set_pos(x, y)
            if (x, y) not in extra_info.keys():
                extra_info[(x, y)] = 'R_' + str(restaurant_id) + "#" + str(len(restaurant.order_list))
            else:
                extra_info[(x, y)] += '\n' + 'R_' + str(restaurant_id) + "#" + str(len(restaurant.order_list))

        for customer_id, customer in self.customers.items():
            x = customer.position[0]
            y = customer.position[1]
            self.images["customers"][customer_id].set_pos(x, y)

        # print("extra info is {}".format(extra_info))

        im_data = np.array(self._render_board(self.grid, GRID_UNIT, FIX, self.images, extra_info))
        self.game_tape.append(im_data)
        return im_data

    def _render_board(self, board, unit, fix, images, extra_info):
        im = board.copy()
        draw = ImageDraw.Draw(im)

        # for road in self.road:
        #     pos = key2pos(road)
        #     row = pos[0]
        #     col = pos[1]
        #     draw.rectangle(build_rectangle(col, row, unit, fix), fill=(250, 235, 215), outline=(0, 0, 0))
        #
        # for edge in self.edges:
        #     pos = key2pos(edge)
        #     row = pos[0]
        #     col = pos[1]
        #     draw.rectangle(build_rectangle(col, row, unit, fix), fill=(250, 128, 114), outline=(0, 0, 0))

        for road in self.horizontal:
            pos = key2pos(road)
            row = pos[0]
            col = pos[1]
            draw.rectangle(build_rectangle(col, row, unit, fix), fill=(220, 20, 60), outline=(0, 0, 0))

        for road in self.vertical:
            pos = key2pos(road)
            row = pos[0]
            col = pos[1]
            draw.rectangle(build_rectangle(col, row, unit, fix), fill=(250, 128, 114), outline=(0, 0, 0))

        for road in self.intersection:
            pos = key2pos(road)
            row = pos[0]
            col = pos[1]
            draw.rectangle(build_rectangle(col, row, unit, fix), fill=(255, 160, 122), outline=(0, 0, 0))

        for image in images["restaurants"]:
            # draw.bitmap((image.y * unit + unit // fix - 4, image.x * unit + unit // fix - 4), image.bitmap, image.color)
            im.paste(image.bitmap, (image.y * unit + unit // fix - 4, image.x * unit + unit // fix - 4))

        for image in images["customers"]:
            # draw.bitmap((image.y * unit + unit // fix - 4, image.x * unit + unit // fix - 4), image.bitmap, image.color)
            im.paste(image.bitmap, (image.y * unit + unit // fix - 4, image.x * unit + unit // fix - 4))

        image_id = 0
        for i in extra_info.keys():
            x = i[0]
            y = i[1]

            value = extra_info[(x, y)]
            values = value.split("\n")

            for v in values:
                if v[0] == 'A':
                    image = images["agents"][image_id]
                    draw.bitmap((image.y * unit + unit // fix - 4, image.x * unit + unit // fix - 4), image.bitmap,
                                image.color)
                    image_id += 1

            draw.text(((y + 1.0 / 20) * unit, (x + 1.0 / 20) * unit), value, fill=(0, 0, 0))
        return im


def agent2dict(agent):
    return {
        "agent_id": copy.deepcopy(agent.agent_id),
        "position": copy.deepcopy(agent.position),
        "order_to_deliver": copy.deepcopy([order2dict(order) for order in agent.order_list]),
        "orders_to_pick": copy.deepcopy([order2dict(order) for order in agent.orders_to_pick])
    }


def agent2dict_info_after(agent):
    return {
        "agent_id": copy.deepcopy(agent.agent_id),
        "position": copy.deepcopy(agent.position),
        "order_to_deliver_length": len(agent.order_list),
        "orders_to_pick_length": len(agent.orders_to_pick)
    }


class Agent:
    def __init__(self, agent_id, init_pos):
        self.agent_id = agent_id
        self.position = init_pos
        self.order_list = []
        self.orders_to_pick = []

    def update_position(self, move_action, board):
        new_row = self.position[0] + DIRECTION[move_action][0]
        new_col = self.position[1] + DIRECTION[move_action][1]

        if 0 <= new_row < N and 0 <= new_col < N and (board[new_row][new_col] == 1 or board[new_row][new_col] == 3
                                                      or board[new_row][new_col] == 4 or board[new_row][new_col] == 5):
            self.position[0] = new_row
            self.position[1] = new_col


def restaurant2dict(restaurant):
    return {
        "restaurant_id": copy.deepcopy(restaurant.restaurant_id),
        "position": copy.deepcopy(restaurant.position),
        "order_list": copy.deepcopy([order2dict(order) for order in restaurant.order_list])
    }


def restaurant2dict_info_after(restaurant):
    return {
        "restaurant_id": copy.deepcopy(restaurant.restaurant_id),
        "position": copy.deepcopy(restaurant.position),
        "order_list_length": len(restaurant.order_list)
    }


class Restaurant:
    def __init__(self, restaurant_id, init_pos):
        self.restaurant_id = restaurant_id
        self.position = init_pos
        self.order_list = []


def customer2dict(customer):
    return {
        "customer_id": copy.deepcopy(customer.customer_id),
        "position": copy.deepcopy(customer.position)
    }


class Customer:
    def __init__(self, customer_id, init_pos):
        self.customer_id = customer_id
        self.position = init_pos


def order2dict(order):
    return {
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "restaurant_id": order.restaurant_id,
        "start_time": order.start_time,
        "end_time": order.end_time,
        "distributed": order.distributed,
        "rider_id": order.rider_id
    }


class Order:
    def __init__(self, order_id, customer_id, restaurant_id, start_time, end_time, distributed, remove,
                 distance):
        self.order_id = order_id
        self.customer_id = customer_id
        self.restaurant_id = restaurant_id
        self.start_time = start_time
        self.end_time = end_time
        self.distributed = distributed
        self.remove = remove
        self.rider_id = -1
        self.distance = distance

    @property
    def to_dict(self):
        return {'order_id': self.order_id,
                "customer_id": self.customer_id,
                "restaurant_id": self.restaurant_id,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "distributed": self.distributed,
                "remove": self.remove,
                "rider_id": self.rider_id}

def check_position(p1, p2):
    return p1[0] == p2[0] and p1[1] == p2[1]


def calculate_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def pos2key(pos):
    return pos[0] * N + pos[1]


def key2pos(key):
    return [key // N, key % N]


def change_background(img, color):
    x, y = img.size
    new_image = Image.new('RGBA', img.size, color=color)
    new_image.paste(img, (0, 0, x, y), img)
    return new_image


def build_rectangle(x, y, unit_size, fix):
    return x * unit_size, y * unit_size, (x + 1) * unit_size, (y + 1) * unit_size


class Bitmap:
    def __init__(self, bitmap, unit, color):
        self.bitmap = bitmap
        self.x = 0
        self.y = 0
        self.unit = unit
        self.reshape()
        self.color = color

    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def reshape(self):
        self.bitmap = self.bitmap.resize((self.unit, self.unit))




COLORS = {
    'red': [255,0,0],
    'light red': [255, 127, 127],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'orange': [255, 127, 0],
    'grey':  [176,196,222],
    'purple': [160, 32, 240],
    'black': [0, 0, 0],
    'white': [255, 255, 255],
    'light green': [204, 255, 229],
    'sky blue': [0,191,255],
    # 'red-2': [215,80,83],
    # 'blue-2': [73,141,247]
}

pygame.init()
font = pygame.font.Font(None, 50)
def debug(info, y = 10, x=10, c='black'):
    display_surf = pygame.display.get_surface()
    debug_surf = font.render(str(info), True, COLORS[c])
    debug_rect = debug_surf.get_rect(topleft = (x,y))
    display_surf.blit(debug_surf, debug_rect)

