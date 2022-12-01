# -*- coding:utf-8  -*-
# Time  : 2022/2/11 下午3:18
# Author: Yahui Cui
import copy
import os
import random
import numpy as np

import pygame
from PIL import Image, ImageDraw

from env.simulators.game import Game

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
        joint_action_decode = [[action[0].index(1) if 1 in action[0] else 4, action[1], action[2]]
                               for action in joint_action]

        return joint_action_decode

    def is_valid_action(self, joint_action):
        if len(joint_action) != self.n_player:
            raise Exception("Input joint action dimension should be {}, not {}.".format(
                self.n_player, len(joint_action)))

        for i in range(self.n_player):
            if len(joint_action[i]) != 3:
                raise Exception("The input action should have 3 parts, not {}".format(len(joint_action[i])))
            if len(joint_action[i][0]) != self.joint_action_space[i][0].n:
                raise Exception("The first action dimension for player {} should be {}, not {}.".format(
                    i, self.joint_action_space[i][0].n, len(joint_action[i][0])))
            if len(joint_action[i][1]) != self.joint_action_space[i][1].n:
                raise Exception("The second action dimension for player {} should be {}, not {}.".format(
                    i, self.joint_action_space[i][1].n, len(joint_action[i][1])))
            if len(joint_action[i][2]) != self.joint_action_space[i][2].n:
                raise Exception("The third action dimension for player {} should be {}, not {}.".format(
                    i, self.joint_action_space[i][2].n, len(joint_action[i][2])))

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
        action_spaces = [[Discrete(5), Discrete(DISTRIBUTE_NUM), Discrete(CAPACITY_RIDER)]
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
CAPACITY_RESTAURANT = 10
RESTAURANT_NUM = 10
CUSTOMER_NUM = 20
TOTAL_STEP = 500
DISTRIBUTE_DISTANCE = 5
DISTRIBUTE_NUM = 20
PICK_UP_TIMES = 2
TIME_INTERVAL_END_TIME = 400

GRID_UNIT = 40
FIX = 8

# 0: obstacles 1: roads 2: available positions for customers and restaurants
# 3: customers and restaurants(modified after init)
MAP = [[0, 0, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 0, 0],
       [0, 0, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 0, 0],
       [2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2],
       [2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2],
       [2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2],
       [2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2],
       [0, 0, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 0, 0],
       [0, 0, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 0, 0]]


class Delivery:
    def __init__(self, agent_num):
        self.agent_num = agent_num
        self.agents = []
        self.customers = {}
        self.restaurants = {}
        self.pos2restaurant = {}
        self.total_order = 0
        self.step_cnt = 0
        # 0: obstacle; 1: road
        self.board = [[0 for _ in range(N)] for _ in range(N)]
        self.road = []
        self.distribute_map = {}
        self.set_up()
        self.step_rewards = [0.0 for _ in range(len(self.agents))]

    def reset(self):
        self.agents = []
        self.customers = {}
        self.restaurants = {}
        self.pos2restaurant = {}
        self.total_order = 0
        self.step_cnt = 0
        self.board = [[0 for _ in range(N)] for _ in range(N)]
        self.road = []
        self.distribute_map = {}
        self.set_up()
        self.step_rewards = [0.0 for _ in range(len(self.agents))]

        current_state = self.get_current_state()
        return current_state

    def set_up(self):
        # add road
        # all_pos_key: add available positions for customers and restaurants
        all_pos_keys = []
        for row in range(N):
            for col in range(N):
                if MAP[row][col] == 1:
                    self.board[row][col] = 1
                    key = pos2key([row, col])
                    self.road.append(key)
                elif MAP[row][col] == 2:
                    self.board[row][col] = 2
                    key = pos2key([row, col])
                    all_pos_keys.append(key)

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
        added_orders_restaurants = [[] for _ in range(len(self.restaurants))]
        for restaurant_id, _ in self.restaurants.items():
            new_orders = self.generate_orders(self.step_cnt, restaurant_id, [])
            added_orders_restaurants[restaurant_id] += new_orders

        for restaurant_id, restaurant in self.restaurants.items():
            restaurant.update_orders([], added_orders_restaurants[restaurant_id])
            restaurant.update_order_numbers()
            self.restaurants[restaurant_id] = restaurant

        self.distribute()

    def step(self, actions):

        self.step_rewards = [0.0 for _ in range(len(self.agents))]
        deleted_orders_agents = [[] for _ in range(len(self.agents))]
        deleted_pick_orders_agents = [[] for _ in range(len(self.agents))]
        deleted_pick_orders_restaurants = [[] for _ in range(len(self.restaurants))]
        deleted_order_ids_restaurants = [[] for _ in range(len(self.restaurants))]

        added_orders_restaurants = [[] for _ in range(len(self.restaurants))]

        move_actions = [action[0] for action in actions]
        pick_order_actions = [action[1] for action in actions]
        put_order_actions = [action[2] for action in actions]

        # update agent position
        for i in range(len(self.agents)):
            agent = self.agents[i]
            agent.update_position(move_actions[i], self.board)

        # check put_order_actions
        for i in range(len(self.agents)):
            agent = self.agents[i]
            put_order_action = put_order_actions[i]
            for j in range(len(agent.order_list)):
                cur_order = agent.order_list[j]
                order_number_in_rider = cur_order.order_number_in_rider
                if order_number_in_rider < 0:
                    raise Exception("order_number_in_rider less than 0!")
                if put_order_action[order_number_in_rider] == 1:
                    deleted_orders_agents[i].append(cur_order)
                    if check_position(agent.position, self.customers[cur_order.customer_id].position) \
                            and cur_order.end_time >= self.step_cnt:
                        self.step_rewards[i] += calculate_distance(self.customers[cur_order.customer_id].position,
                                                                   self.restaurants[cur_order.restaurant_id].position)
                    else:
                        self.step_rewards[i] -= 100
                elif cur_order.end_time < self.step_cnt:
                    deleted_orders_agents[i].append(cur_order)
                    self.step_rewards[i] -= 100

        # check orders_to_pick
        for i in range(len(self.agents)):
            agent = self.agents[i]
            for order_to_pick in agent.orders_to_pick:
                if order_to_pick.pick_up_time < self.step_cnt:
                    self.step_rewards[i] -= 100
                    deleted_pick_orders_agents[i].append(order_to_pick)
                    # if the order is out of pick up time, the order will be reassigned.
                    deleted_pick_orders_restaurants[order_to_pick.restaurant_id].append(order_to_pick)
                # if the order is out of end_time, drop the order
                elif order_to_pick.end_time < self.step_cnt:
                    self.step_rewards[i] -= 100
                    deleted_pick_orders_agents[i].append(order_to_pick)
                    deleted_order_ids_restaurants[order_to_pick.restaurant_id].append(order_to_pick.order_id)

        added_pick_orders_agents, added_pick_orders_restaurants, deleted_order_ids_restaurants, single_rewards = \
            self.accept_orders(pick_order_actions, deleted_orders_agents, deleted_pick_orders_agents,
                               deleted_order_ids_restaurants)

        for i in range(len(self.agents)):
            self.step_rewards[i] += single_rewards[i]

        added_orders_agents, added_pick_orders_agents, added_pick_orders_restaurants, deleted_pick_orders_agents, \
        deleted_order_ids_restaurants = self.pick_up_orders(deleted_pick_orders_agents, added_pick_orders_agents,
                                                            added_pick_orders_restaurants,
                                                            deleted_order_ids_restaurants)

        for restaurant_id, restaurant in self.restaurants.items():
            for order in restaurant.order_list:
                if order.end_time < self.step_cnt and order.order_id not in \
                        deleted_order_ids_restaurants[restaurant_id]:
                    deleted_order_ids_restaurants[restaurant_id].append(order.order_id)

        # generate new orders for restaurants
        if self.step_cnt % UPDATE_FREQUENCY == 0 and self.step_cnt > 0:
            for restaurant_id, _ in self.restaurants.items():
                new_orders = self.generate_orders(self.step_cnt, restaurant_id,
                                                  deleted_order_ids_restaurants[restaurant_id])
                added_orders_restaurants[restaurant_id] += new_orders

        # update agent information
        for i in range(len(self.agents)):
            agent = self.agents[i]
            agent.update_orders(deleted_orders_agents[i], added_orders_agents[i])
            agent.update_order_numbers()
            agent.update_orders_to_pick(deleted_pick_orders_agents[i], added_pick_orders_agents[i])
            order_set = set([order.order_id for order in agent.order_list])
            order_to_pick_set = set([order.order_id for order in agent.orders_to_pick])
            intersection = order_set.intersection(order_to_pick_set)
            if len(intersection) > 0:
                raise Exception("Agent order list and order list to pick has intersection! ", intersection)

        # update restaurant information
        for restaurant_id, restaurant in self.restaurants.items():
            restaurant.update_orders(deleted_order_ids_restaurants[restaurant_id],
                                     added_orders_restaurants[restaurant_id])
            restaurant.update_order_numbers()
            restaurant.update_pick_up_order_status(deleted_pick_orders_restaurants[restaurant_id],
                                                   added_pick_orders_restaurants[restaurant_id])
            self.restaurants[restaurant_id] = restaurant

        self.distribute()

        current_state = self.get_current_state()

        self.step_cnt += 1
        done = self.step_cnt >= TOTAL_STEP

        if done:
            # when done, if there are still orders to be picked up or ordered hasn't been sent
            for i in range(len(self.agents)):
                agent = self.agents[i]
                self.step_rewards[i] -= 100 * len(agent.orders_to_pick)
                self.step_rewards[i] -= 100 * len(agent.order_list)

        info = self.get_info_after()

        return current_state, self.step_rewards, done, info

    def generate_orders(self, step_cnt, restaurant_id, deleted_orders_restaurant):
        restaurant = self.restaurants[restaurant_id]
        new_orders = []
        while len(restaurant.order_list) - len(deleted_orders_restaurant) + len(new_orders) < CAPACITY_RESTAURANT:
            order_id = self.total_order
            customer_id = random.randint(0, len(self.customers) - 1)
            start_time = step_cnt
            distance = calculate_distance(restaurant.position, self.customers[customer_id].position)
            end_time = random.randint(step_cnt + distance, step_cnt + TIME_INTERVAL_END_TIME)
            new_order = Order(order_id, customer_id, restaurant_id, start_time, end_time, -1)
            new_orders.append(new_order)
            self.total_order += 1

        return new_orders

    def distribute(self):
        # distribute orders in restaurant to riders whose distance < DISTRIBUTE_DISTANCE.
        # Each rider can be distributed up to 20 orders.
        self.distribute_map = {}
        for agent in self.agents:
            self.distribute_map[agent.agent_id] = []
            restaurants_to_sort = list(self.restaurants.values()).copy()
            new_restaurants_to_sort = sorted(restaurants_to_sort,
                                             key=lambda restaurant: calculate_distance(restaurant.position,
                                                                                       agent.position))
            for restaurant in new_restaurants_to_sort:
                if calculate_distance(restaurant.position, agent.position) < DISTRIBUTE_DISTANCE:
                    order_index = 0
                    while len(self.distribute_map[agent.agent_id]) < DISTRIBUTE_NUM \
                            and order_index < len(restaurant.order_list):
                        if restaurant.order_list[order_index].distributed == -1:
                            self.distribute_map[agent.agent_id].append(restaurant.order_list[order_index])
                        order_index += 1
                else:
                    break

        for key, val in self.distribute_map.items():
            new_val = sorted(val, key=lambda order: order.order_id)
            self.distribute_map[key] = new_val

    def accept_orders(self, pick_order_actions, deleted_orders_agents, deleted_pick_orders_agents,
                      deleted_order_ids_restaurants):
        single_rewards = [0.0 for _ in range(len(self.agents))]
        added_pick_orders_agents_temp = [[] for _ in range(len(self.agents))]
        added_pick_orders_agents = [[] for _ in range(len(self.agents))]
        added_pick_orders_restaurants = [[] for _ in range(len(self.restaurants))]

        order_id2riders = {}
        order_id2order = {}
        all_order_ids = []
        all_orders = []
        for i in range(len(self.agents)):
            agent = self.agents[i]
            pick_order_action = pick_order_actions[i]
            allocated_orders = self.distribute_map[agent.agent_id]
            for j in range(len(allocated_orders)):
                cur_order = allocated_orders[j]
                order_id2order[cur_order.order_id] = cur_order
                if pick_order_action[j] == 1:
                    if cur_order.order_id not in order_id2riders.keys():
                        order_id2riders[cur_order.order_id] = [agent.agent_id]
                    else:
                        order_id2riders[cur_order.order_id].append(agent.agent_id)
                    if cur_order.order_id not in all_order_ids:
                        all_order_ids.append(cur_order.order_id)
                        all_orders.append(cur_order)

        # allocate orders to riders according to order_id2riders
        for key, value in order_id2riders.items():
            if len(value) > 1:
                random.shuffle(value)
                order_id2riders[key] = value

        new_all_orders = sorted(all_orders, key=lambda single_order: single_order.order_id)

        for order in new_all_orders:
            selected_agents = order_id2riders[order.order_id]
            for selected_agent_id in selected_agents:
                if len(self.agents[selected_agent_id].order_list) + len(self.agents[selected_agent_id].orders_to_pick) \
                        + len(added_pick_orders_agents_temp[selected_agent_id]) - \
                        len(deleted_orders_agents[selected_agent_id]) - \
                        len(deleted_pick_orders_agents[selected_agent_id]) < CAPACITY_RIDER:
                    order_ = order_id2order[order.order_id]
                    added_pick_orders_agents_temp[selected_agent_id].append(order_)
                    break

        for i in range(len(self.agents)):
            selected_agent_id = i
            added_pick_orders_agents_step = added_pick_orders_agents_temp[selected_agent_id]
            for added_order in added_pick_orders_agents_step:
                if added_order.end_time >= self.step_cnt:
                    distance_to_restaurant = calculate_distance(self.agents[selected_agent_id].position,
                                                                self.restaurants[added_order.restaurant_id].position)
                    pick_up_time = random.randint(distance_to_restaurant, PICK_UP_TIMES * distance_to_restaurant)
                    # set the pick_up_time for the order when the order is accepted.
                    added_order.set_pick_up_time(self.step_cnt + pick_up_time)
                    added_order.distributed = selected_agent_id
                    added_pick_orders_agents[selected_agent_id].append(added_order)
                    added_pick_orders_restaurants[added_order.restaurant_id].append(added_order)
                else:
                    single_rewards[selected_agent_id] -= 100
                    if added_order.order_id not in deleted_order_ids_restaurants[added_order.restaurant_id]:
                        deleted_order_ids_restaurants[added_order.restaurant_id].append(added_order.order_id)

        return added_pick_orders_agents, added_pick_orders_restaurants, deleted_order_ids_restaurants, single_rewards

    def pick_up_orders(self, deleted_pick_orders_agents, added_pick_orders_agents, added_pick_orders_restaurants,
                       deleted_order_ids_restaurants):
        # if agent is at restaurant, check his/her orders_to_pick
        added_orders_agents = [[] for _ in range(len(self.agents))]

        for i in range(len(self.agents)):
            agent = self.agents[i]
            pos_key = pos2key(agent.position)
            if pos_key in self.pos2restaurant.keys():
                restaurant_id = self.pos2restaurant[pos_key]
                # orders have already picked
                for cur_order in agent.orders_to_pick:
                    if cur_order.restaurant_id == restaurant_id and cur_order.pick_up_time >= self.step_cnt \
                            and cur_order.end_time >= self.step_cnt:
                        added_orders_agents[agent.agent_id].append(cur_order)
                        deleted_pick_orders_agents[agent.agent_id].append(cur_order)
                        if cur_order.order_id not in deleted_order_ids_restaurants[cur_order.restaurant_id]:
                            deleted_order_ids_restaurants[cur_order.restaurant_id].append(cur_order.order_id)
                # orders picked in this step
                picked_orders_step = added_pick_orders_agents[i]
                added_pick_orders_agents_to_remove = []
                for cur_order in picked_orders_step:
                    if cur_order.restaurant_id == restaurant_id and cur_order.pick_up_time >= self.step_cnt \
                            and cur_order.end_time >= self.step_cnt:
                        added_pick_orders_agents_to_remove.append(cur_order)
                        added_orders_agents[agent.agent_id].append(cur_order)
                        if cur_order.order_id not in deleted_order_ids_restaurants[cur_order.restaurant_id]:
                            deleted_order_ids_restaurants[cur_order.restaurant_id].append(cur_order.order_id)
                for order_to_remove in added_pick_orders_agents_to_remove:
                    added_pick_orders_agents[i].remove(order_to_remove)
                for order_to_remove in added_pick_orders_agents_to_remove:
                    added_pick_orders_restaurants_to_move = []
                    for order in added_pick_orders_restaurants[order_to_remove.restaurant_id]:
                        if order.order_id == order_to_remove.order_id:
                            added_pick_orders_restaurants_to_move.append(order)
                    for order in added_pick_orders_restaurants_to_move:
                        added_pick_orders_restaurants[order_to_remove.restaurant_id].remove(order)

        return added_orders_agents, added_pick_orders_agents, added_pick_orders_restaurants, \
               deleted_pick_orders_agents, deleted_order_ids_restaurants

    def get_current_state(self):
        return {
            "agents": [agent2dict(agent) for agent in self.agents],
            "restaurants": [restaurant2dict(restaurant) for _, restaurant in self.restaurants.items()],
            "customers": [customer2dict(customer) for _, customer in self.customers.items()],
            "distributed_orders": copy.deepcopy([(agent_id, [order2dict(order) for order in agent_orders])
                                                 for agent_id, agent_orders in self.distribute_map.items()]),
            "roads": copy.deepcopy([[key2pos(pos)[0], key2pos(pos)[1]] for pos in self.road])
        }

    def get_info_after(self):
        return {
            "agents": [agent2dict_info_after(agent) for agent in self.agents],
            "restaurants": [restaurant2dict_info_after(restaurant) for _, restaurant in self.restaurants.items()],
            "customers": [customer2dict(customer) for _, customer in self.customers.items()],
            "distributed_orders": copy.deepcopy([(agent_id, [order2dict(order) for order in agent_orders])
                                                 for agent_id, agent_orders in self.distribute_map.items()]),
            "step_rewards": self.step_rewards
        }

    def get_init_info(self):
        return {
            "agents": [agent2dict_info_after(agent) for agent in self.agents],
            "restaurants": [restaurant2dict_info_after(restaurant) for _, restaurant in self.restaurants.items()],
            "customers": [customer2dict(customer) for _, customer in self.customers.items()],
            "roads": copy.deepcopy([[key2pos(pos)[0], key2pos(pos)[1]] for pos in self.road]),
            "distributed_orders": copy.deepcopy([(agent_id, [order2dict(order) for order in agent_orders])
                                                 for agent_id, agent_orders in self.distribute_map.items()])
        }

    def render(self, fps=1):
        if self.step_cnt == 0:
            # images
            resource_path_rider = os.path.join(os.path.dirname(__file__), "images", "delivery", "rider.png")
            resource_path_restaurant = os.path.join(os.path.dirname(__file__), "images", "delivery", "restaurant.png")
            resource_path_customer = os.path.join(os.path.dirname(__file__), "images", "delivery", "customer.png")
            self.images = {
                "agents": [Bitmap(Image.open(resource_path_rider), GRID_UNIT, (0, 191, 255))
                           for _ in range(len(self.agents))],
                "restaurants": [Bitmap(change_background(Image.open(resource_path_restaurant), (148, 0, 211)),
                                       GRID_UNIT - 1, (0, 0, 0)) for _ in range(len(self.restaurants))],
                "customers": [
                    Bitmap(change_background(Image.open(resource_path_customer), (148, 0, 211)), GRID_UNIT - 1,
                           (0, 0, 0)) for _ in range(len(self.customers))]
            }
            pygame.init()
            self.grid = Delivery.init_board(N, N, GRID_UNIT)
            self.screen = pygame.display.set_mode(self.grid.size)
            # pygame.display.set_caption(self.game_name)
            self.game_tape = []

            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        pygame.surfarray.blit_array(self.screen, self.render_board().transpose(1, 0, 2))
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

        for road in self.road:
            pos = key2pos(road)
            row = pos[0]
            col = pos[1]
            draw.rectangle(build_rectangle(col, row, unit, fix), fill=(250, 235, 215), outline=(0, 0, 0))

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
        "order_list": copy.deepcopy([order2dict(order) for order in agent.order_list]),
        "orders_to_pick": copy.deepcopy([order2dict(order) for order in agent.orders_to_pick])
    }


def agent2dict_info_after(agent):
    return {
        "agent_id": copy.deepcopy(agent.agent_id),
        "position": copy.deepcopy(agent.position),
        "order_list_length": len(agent.order_list)
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

        if 0 <= new_row < N and 0 <= new_col < N and (board[new_row][new_col] == 1 or board[new_row][new_col] == 3):
            self.position[0] = new_row
            self.position[1] = new_col

    def update_orders(self, orders_to_remove, orders_to_add):
        delete_order_ids = []
        delete_orders = []

        for order in orders_to_remove:
            delete_order_ids.append(order.order_id)

        for cur_order in self.order_list:
            if cur_order.order_id in delete_order_ids:
                delete_orders.append(cur_order)

        for order in delete_orders:
            self.order_list.remove(order)

        for order in orders_to_add:
            self.order_list.append(order)

    def update_order_numbers(self):
        new_order_list = []
        for i in range(len(self.order_list)):
            order = copy.deepcopy(self.order_list[i])
            order.order_number_in_rider = i
            new_order_list.append(order)

        self.order_list = new_order_list

    def update_orders_to_pick(self, orders_to_remove, orders_to_add):
        delete_order_ids = []
        delete_orders = []

        for order in orders_to_remove:
            delete_order_ids.append(order.order_id)

        for cur_order in self.orders_to_pick:
            if cur_order.order_id in delete_order_ids:
                delete_orders.append(cur_order)

        for order in delete_orders:
            self.orders_to_pick.remove(order)

        for order in orders_to_add:
            self.orders_to_pick.append(order)


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

    def update_orders(self, order_ids_to_remove, orders_to_add):
        delete_order_ids = []
        delete_orders = []
        for order_id in order_ids_to_remove:
            delete_order_ids.append(order_id)

        for cur_order in self.order_list:
            if cur_order.order_id in delete_order_ids:
                delete_orders.append(cur_order)

        for order in delete_orders:
            self.order_list.remove(order)

        for order in orders_to_add:
            self.order_list.append(order)

    def update_order_numbers(self):
        new_order_list = []
        for i in range(len(self.order_list)):
            order = copy.deepcopy(self.order_list[i])
            order.order_number_in_restaurant = i
            new_order_list.append(order)

        self.order_list = new_order_list

    def update_pick_up_order_status(self, deleted_pick_orders_restaurant, added_pick_order_restaurant):
        new_order_map = {}
        new_deleted_order_list = []

        for order in added_pick_order_restaurant:
            new_order_map[order.order_id] = order

        for order in deleted_pick_orders_restaurant:
            new_deleted_order_list.append(order.order_id)

        for order in self.order_list:
            if order.order_id in new_order_map.keys():
                order.distributed = new_order_map[order.order_id].distributed
                order.pick_up_time = new_order_map[order.order_id].pick_up_time

            if order.order_id in new_deleted_order_list:
                order.distributed = -1
                order.pick_up_time = 0
                order.order_number_in_rider = -1


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
        "pick_up_time": order.pick_up_time,
        "distributed": order.distributed,
        "order_number_in_restaurant": order.order_number_in_restaurant,
        "order_number_in_rider": order.order_number_in_rider
    }


class Order:
    def __init__(self, order_id, customer_id, restaurant_id, start_time, end_time, distributed):
        self.order_id = order_id
        self.customer_id = customer_id
        self.restaurant_id = restaurant_id
        self.start_time = start_time
        self.end_time = end_time
        self.pick_up_time = 0
        self.distributed = distributed
        self.order_number_in_restaurant = -1
        self.order_number_in_rider = -1

    def set_pick_up_time(self, pick_up_time):
        self.pick_up_time = pick_up_time


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
