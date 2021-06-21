# -*- coding:utf-8  -*-
# Time  : 2021/04/02 17:07
# Author: Yutong Wu

import torch
from tensorboardX import SummaryWriter
from collections import namedtuple
from itertools import count
from common.log_path import make_logpath

from agent import ini_agents
from common.utils import *
import numpy as np
import random
import os

class Runner:
    def __init__(self, args, env):

        self.args = args

        # torch.manual_seed(self.args.seed_nn)
        # np.random.seed(self.args.seed_np)
        # random.seed(self.args.seed_random)

        self.env = env
        self.agent = ini_agents(args)

        # 保存训练参数 以便复现
        if self.args.reload_config:
            file_name = self.args.algo + "_" + self.args.scenario
            self.args = config_dir = os.path.join(os.getcwd(), "config")
            self.args = load_config(self.args, config_dir, file_name)
        else: # todo: 重复处理
            file_name = self.args.algo + "_" + self.args.scenario
            config_dir = os.path.join(os.getcwd(), "config")
            save_config(self.args, config_dir, file_name=file_name)

    def set_up(self):

        # 设置seed, 以便复现
        # torch.manual_seed(self.args.seed_nn)
        # np.random.seed(self.args.seed_np)
        # random.seed(self.args.seed_random)

        # 定义保存路径
        run_dir, log_dir = make_logpath(self.args.scenario, self.args.algo)
        self.writer = SummaryWriter(str(log_dir))

    def run(self):

        self.set_up()

        for i_epoch in range(self.args.max_episodes):
            state = self.env.reset()
            Gt = 0
            for t in count():

                action = self.agent.choose_action(state)

                next_state, reward, done, _, _ = self.env.step(action)

                self.agent.memory.push((state, next_state, action, reward, np.float32(done)))

                self.agent.learn()

                state = next_state

                Gt += reward

                if done:
                    print('i_epoch: ', i_epoch, 'Gt: ', '%.2f' % Gt, 'epi: ', '%.2f' % self.agent.eps)
                    reward_tag = 'reward'
                    self.writer.add_scalars(reward_tag, global_step=i_epoch,
                                       tag_scalar_dict={'return': Gt})

                    if i_epoch % self.args.evaluate_rate == 0 and i_epoch > 1:
                        Gt_real = self.evaluate(i_epoch)
                        self.writer.add_scalars(reward_tag, global_step=i_epoch, tag_scalar_dict={'real_return': Gt_real})
                    break

    def evaluate(self, i_epoch):
        record = []
        for _ in range(10):
            state = self.env.reset()
            Gt_real = 0
            for t in count():
                action = self.agent.choose_action(state, train=False)
                next_state, reward, done, _, _ = self.env.step(action, train=False)
                state = next_state
                Gt_real += reward
                if done:
                    record.append(Gt_real)
                    break
        print('===============', 'i_epoch: ', i_epoch, 'Gt_real: ', '%.2f' % np.mean(record))
        return np.mean(record)
