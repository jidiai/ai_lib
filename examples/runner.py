# -*- coding:utf-8  -*-
# Time  : 2021/04/02 17:07
# Author: Yutong Wu

from tensorboardX import SummaryWriter
from itertools import count
from common.log_path import make_logpath

from agent import ini_agents
from common.utils import *

from common.settings import *

import torch
import numpy as np
import random


class Runner:
    def __init__(self, args):

        env = make_env(args)
        self.env = env
        self.EnvSetting = EnvSettingDefault(
            scenario=args.scenario,
            action_space=self.env.get_actionspace(),
            obs_space=self.env.get_observationspace())

        self.run_dir, self.log_dir = make_logpath(args.scenario, args.algo)
        self.writer = SummaryWriter(str(self.run_dir))

        config_dir = os.path.join(os.getcwd(), "config")
        file_name = args.algo + "_" + args.scenario

        if (not args.reload_config and not os.path.exists(os.path.join(self.log_dir, file_name + '.yaml'))) \
                or (args.reload_config and not os.path.exists(os.path.join(config_dir, file_name + '.yaml'))):
            paras = TrainerSettings(
                algo=args.algo,
                hyperparameters=globals()[str(args.algo).upper() + "Settings"](),
                envparameters=self.EnvSetting,
                trainingparameters=TrainingDefault(),
                seedparameters=SeedSetting())
            save_new_paras(paras, self.log_dir, file_name)
            config_dict = load_config(self.log_dir, file_name)
        elif not args.reload_config and os.path.exists(os.path.join(self.log_dir, file_name + '.yaml')):
            config_dict = load_config(self.log_dir, file_name)
        else:
            config_dict = load_config(config_dir, file_name)

        paras = get_paras_from_dict(config_dict)
        self.paras = paras

        save_config(config_dict, self.log_dir, file_name=file_name)

        torch.manual_seed(self.paras.seed_nn)
        np.random.seed(self.paras.seed_np)
        random.seed(self.paras.seed_random)

        self.agent = ini_agents(self.paras)

    def add_experience(self, states, state_next, reward, done):
        agent_id = 0
        self.agent.memory.insert("states", agent_id, states)
        self.agent.memory.insert("states_next", agent_id, state_next)
        self.agent.memory.insert("rewards", agent_id, reward)
        self.agent.memory.insert("dones", agent_id, np.array(done, dtype=bool))

    def run(self):

        for i_epoch in range(self.paras.max_episodes):
            self.env.set_seed(random.randint(0, sys.maxsize))
            state = self.env.reset()
            Gt = 0
            for t in count():

                if self.paras.render:
                    self.env.make_render()

                action = self.agent.choose_action(state, train=True)

                next_state, reward, done, _, _ = self.env.step(action)

                self.add_experience(state, next_state, reward, np.float32(done))

                state = next_state

                Gt += reward

                if t % self.paras.learn_freq == 0 and not self.paras.learn_terminal:
                    self.agent.learn()

                if done:
                    if self.paras.learn_terminal:
                        self.agent.learn()
                    print('i_epoch: ', i_epoch, 'Gt: ', '%.2f' % Gt)
                    reward_tag = 'reward'
                    self.writer.add_scalars(reward_tag, global_step=i_epoch,
                                            tag_scalar_dict={'return': Gt})

                    if i_epoch % self.paras.evaluate_rate == 0 and i_epoch > 1:
                        Gt_real = self.evaluate(i_epoch)
                        self.writer.add_scalars(reward_tag, global_step=i_epoch,
                                                tag_scalar_dict={'real_return': Gt_real})
                    break

    def evaluate(self, i_epoch):
        record = []
        for _ in range(10):
            self.env.set_seed(random.randint(0, sys.maxsize))
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
