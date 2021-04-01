# -*- coding:utf-8  -*-
# Time  : 2021/02/25 18:46
# Author: Yutong Wu

import os

import torch
import numpy as np
import argparse
import datetime

from tensorboardX import SummaryWriter
from collections import namedtuple
from itertools import count
from pathlib import Path
import sys

from ppo.ppo import PPO

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from env.chooseenv import make

def action_wrapper(joint_action):
    '''
    :param joint_action:
    :return: wrapped joint action: one-hot
    '''
    joint_action_ = [[0] * 15 for c in range(2)]
    joint_action_fillin = [joint_action_ for m in range(2)]
    for n in range(len(joint_action)):
        cnt = 0
        for i in range(15):
            for j in range(15):
                if cnt == joint_action[n]:
                    joint_action_fillin[n][0][i] = 1
                    joint_action_fillin[n][1][j] = 1
                    return joint_action_fillin
                else: cnt += 1

def state_wrapper(obs):
    '''
    :param state:
    :return: wrapped state
    '''
    obs_ = []
    for i in range(env.board_height):
        for j in range(env.board_width):
            obs_.append(obs[i][j][0])
    # # TODO hard code
    # for n in game.snakes_position:
    #     obs_.append(game.snakes_position[n][0][0])
    return obs_

def main(args):
    print('============ start ============')

    game_name = args.scenario
    global env
    env = make(game_name)
    print('env', env)
    model = args.model

    action_space = env.action_dim
    observation_space = env.input_dimension

    print(f'action_space: {action_space}')
    print(f'observation_space: {observation_space}')

    torch.manual_seed(args.seed)
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

    if args.algo == "ppo":
        agent = PPO(observation_space, action_space, args)

    if model:
        model.load_model()
        print(model)

    episode = 0
    total_step = 0

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / Path('./models') / game_name

    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir #/ 'logs'
    #os.makedirs(log_dir)

    if args.tensorboard and args.mode == "train":
        writer = SummaryWriter(str(log_dir))

    for i_epoch in range(args.max_episode):
        print(i_epoch)
        state = env.reset()

        episode += 1
        step = 0

        for t in count():
            state = np.array(state_wrapper(state))
            # print('state', state.shape, type(state))
            action, action_prob = agent.choose_action(state)
            action_ = action_wrapper([action])
            next_state, reward, done, info, _ = env.step(action_)

            step += 1
            total_step += 1
            reward = np.array(reward[0])

            # print('state', state, 'next_state', next_state)
            # print('state', state.shape, type(state))
            # print('action', action, type(action))
            # print('reward', reward, type(reward))

            trans = Transition(state, action,
                               action_prob, reward, next_state[0])
            agent.store_transition(trans)
            state = next_state

            if done:
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_epoch)
                writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break
    agent.save(run_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="gobang_1v1", type=str)
    parser.add_argument('--max_episode', default=10, type=int)
    parser.add_argument('--algo', default="ppo", type=str, help="dqn/ppo/a2c")

    parser.add_argument('--buffer_capacity', default=int(1e5), type=int)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--clip_param', default=0.2, type=int)
    parser.add_argument('--max_grad_norm', default=0.5, type=int)
    parser.add_argument('--ppo_update_time', default=10, type=int)
    parser.add_argument('--gamma', default=0.99)

    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument('--model', default=None)
    parser.add_argument('--mode', default="train")

    args = parser.parse_args()
    main(args)
