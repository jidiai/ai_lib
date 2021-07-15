# -*- coding:utf-8  -*-
# Time  : 2021/02/26 16:18
# Author: Yutong Wu
import torch
import argparse

from tensorboardX import SummaryWriter
from collections import namedtuple
from itertools import count
from pathlib import Path
import sys

from game_user.examples.common.log_path import make_logpath

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make


def main(args):
    print('============ start ============')

    # 定义环境
    game_name = args.scenario
    global env
    env = make(game_name)

    # 定义wrapper
    # wrapper = eval(str(game_name) + '_wrapper')
    wrapper = globals()[str(game_name).replace('-', '_') + '_wrapper']
    W = wrapper(env)

    # 定义action space和state space
    action_space = W.action_space()
    observation_space = W.state_space()

    # 定义agent
    Agent = globals()[str(args.algo)]
    torch.manual_seed(args.seed)

    print('env', env)
    print('agent', Agent)
    print(f'action_space: {action_space}')
    print(f'observation_space: {observation_space}')

    agent = Agent(observation_space, action_space, args)

    # TODO: transition需要重新定义
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

    # 定义保存路径
    run_dir, log_dir = make_logpath(game_name)

    if args.tensorboard and args.mode == "train":
        writer = SummaryWriter(str(log_dir))

    # 开始训练
    for i_epoch in range(args.max_episode):
        Gt = 0
        obs = env.reset()
        for t in count():
            obs = W.observation(obs, t)
            # todo: 产生两个值，怎么根据不同算法去存?
            action, action_prob = agent.choose_action(obs)
            action_ = W.action(action)
            obs_, reward, done, info, _ = env.step(action_)
            reward = W.reward(reward)

            trans = Transition(obs, action_, action_prob, reward, obs_)
            agent.store_transition(trans)
            obs = obs_

            Gt += reward
            # todo: 检查done， 和每个环境is_terminal相对应
            if done:
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_epoch)
                # print('t: ', t, "return: ", Gt)
                writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break
    agent.save(run_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="classic_CartPole-v0", type=str)
    parser.add_argument('--max_episode', default=500, type=int)
    parser.add_argument('--algo', default="PPO", type=str, help="dqn/PPO/a2c/ddqn")

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