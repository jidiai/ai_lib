# -*- coding:utf-8  -*-
# Time  : 2021/04/02 17:07
# Author: Yutong Wu

from tensorboardX import SummaryWriter
from itertools import count
from common.log_path import make_logpath

# from agents import ini_agents
# todo
from agents.singleagent import SingleRLAgent
from agents.multiagents import MultiRLAgents
from common.utils import *

from common.settings import *

import torch
import numpy as np
import random


class Runner:
    def __init__(self, args):

        env = make_env(args)
        self.env = env
        self.g_core = self.env.env
        self.EnvSetting = EnvSettingDefault(
            scenario=args.scenario,
            action_space=self.env.get_actionspace(),
            obs_space=self.env.get_observationspace(),
            n_player=self.g_core.n_player
        )

        self.run_dir, self.log_dir = make_logpath(args.scenario, args.algo)
        self.writer = SummaryWriter(str(self.run_dir))

        config_dir = os.path.join(os.getcwd(), "config")
        file_name = args.algo + "_" + args.scenario

        if (not args.reload_config and not os.path.exists(os.path.join(self.log_dir, file_name + '.yaml'))) \
                or (args.reload_config and not os.path.exists(os.path.join(config_dir, file_name + '.yaml')) and
                not os.path.exists(os.path.join(self.log_dir, file_name + '.yaml'))):
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

        elif (args.reload_config and not os.path.exists(os.path.join(config_dir, file_name + '.yaml')) and
                os.path.exists(os.path.join(self.log_dir, file_name + '.yaml'))):
            config_dict = load_config(self.log_dir, file_name)

        else:
            config_dict = load_config(config_dir, file_name)

        paras = get_paras_from_dict(config_dict)
        self.paras = paras

        save_config(config_dict, self.run_dir, file_name=file_name)

        torch.manual_seed(self.paras.seed_nn)
        np.random.seed(self.paras.seed_np)
        random.seed(self.paras.seed_random)

        # todo
        if self.paras.marl:
            self.agent = MultiRLAgents(self.paras)
        else:
            self.agent = SingleRLAgent(self.paras)
        self.policy = [paras.algo]
        self.agent_num = 1

    def add_experience(self, states, state_next, reward, done):
        for agent_index, agent_i in enumerate(self.agent.agent):
            agent_i.memory.insert("states", agent_index, states[agent_index]["obs"])
            agent_i.memory.insert("states_next", agent_index, state_next[agent_index]["obs"])
            agent_i.memory.insert("rewards", agent_index, reward)
            agent_i.memory.insert("dones", agent_index, np.array(done, dtype=bool))

    def get_players_and_action_space_list(self):
        if sum(self.g_core.agent_nums) != self.g_core.n_player:
            raise Exception("agents number = %d 不正确，与n_player = %d 不匹配" % (sum(self.g_core.agent_nums), self.g_core.n_player))

        n_agent_num = list(self.g_core.agent_nums)

        for i in range(1, len(n_agent_num)):
            n_agent_num[i] += n_agent_num[i - 1]

        players_id = []
        actions_space = []
        for policy_i in range(len(self.g_core.obs_type)):
            if policy_i == 0:
                players_id_list = range(n_agent_num[policy_i])
            else:
                players_id_list = range(n_agent_num[policy_i - 1], n_agent_num[policy_i])
            players_id.append(players_id_list)

            action_space_list = [self.g_core.get_single_action_space(player_id) for player_id in players_id_list]
            actions_space.append(action_space_list)

        return players_id, actions_space

    # ==========================================================================================================
    # ============================ inference ==================================
    def get_joint_action_eval(self, game, multi_part_agent_ids, policy_list, actions_spaces, all_observes):
        joint_action = []
        for policy_i in range(len(policy_list)):
            agents_id_list = multi_part_agent_ids[policy_i]
            action_space_list = actions_spaces[policy_i]
            function_name = 'm%d' % policy_i
            for i in range(len(agents_id_list)):
                agent_id = agents_id_list[i]
                a_obs = all_observes[agent_id]
                each = self.agent.choose_action_to_env(a_obs)
                joint_action.append(each)
        return joint_action

    def run(self):

        multi_part_agent_ids, actions_space = self.get_players_and_action_space_list()

        for i_epoch in range(1, self.paras.max_episodes+1):
            self.env.set_seed(random.randint(0, sys.maxsize))
            state = self.env.reset()
            step = 0
            Gt = 0
            while not self.g_core.is_terminal():
                step += 1
                joint_act = self.get_joint_action_eval(self.env, multi_part_agent_ids, self.policy, actions_space, state)
                next_state, reward, done, info_before, info_after = self.env.step(joint_act)
                self.add_experience(state, next_state, reward, np.float32(done))

                state = next_state
                if self.paras.marl:
                    reward = sum(reward)
                Gt += reward
                if not self.paras.learn_terminal:
                    if step % self.paras.learn_freq == 0:
                        self.agent.learn()

            if self.paras.learn_terminal:
                self.agent.learn()
            print('i_epoch: ', i_epoch, 'Gt: ', '%.2f' % Gt)
            reward_tag = 'reward'
            self.writer.add_scalars(reward_tag, global_step=i_epoch,
                                    tag_scalar_dict={'return': Gt})

            if i_epoch % self.paras.save_interval == 0:
                self.agent.save(self.run_dir, i_epoch)

            # if i_epoch % self.paras.evaluate_rate == 0 and i_epoch > 1:
            #     Gt_real = self.evaluate(i_epoch)
            #     self.writer.add_scalars(reward_tag, global_step=i_epoch,
            #                             tag_scalar_dict={'real_return': Gt_real})


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
