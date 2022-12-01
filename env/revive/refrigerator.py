# -*- coding:utf-8  -*-
# Time  : 2022/7/25 下午5:33
# Author: Yahui Cui

import random
import numpy as np


class DoorOpen():
    """The agent that controls the opening and closing of the refrigerator door."""

    def __init__(self, door_open_time=10):
        self.door_open_time = door_open_time
        self.init_door_open_time = door_open_time
        self.door_open = False

    def act(self):
        self.door_open_time -= 1
        if self.door_open_time >= 0:
            self.door_open = True
        else:
            self.door_open = False
        return self.door_open

    def reset(self):
        self.door_open = False
        self.door_open_time = self.init_door_open_time


class Simulator:
    """Refrigerator temperature control simulator."""

    def __init__(self, init_temperature=10):
        self.outdoor_temperature = 15
        self.temp = init_temperature
        self.door_state = False

    def update(self, power, dt, door_open=False):
        self.door_state = door_open

        if power > 0:
            self.temp -= power * dt
        if self.door_state == False:
            self.temp = self.temp - (self.temp - self.outdoor_temperature) * 0.02 * dt
        else:
            self.temp = self.temp - (self.temp - self.outdoor_temperature) * 0.08 * dt
        return self.get_temperature()

    def get_temperature(self):
        return self.temp + np.random.normal(0, 0.1)

    def get_door_state(self):
        return self.door_state

    def reset(self, init_temperature):
        self.temp = init_temperature
        self.outdoor_temperature = 15
        random.seed(0)
        np.random.seed(0)
