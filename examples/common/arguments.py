import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # set env and algo
    parser.add_argument('--scenario', default="classic_CartPole-v0", type=str)
    parser.add_argument('--max_episodes', default=5000, type=int)
    parser.add_argument('--algo', default="ppo", type=str, help="dqn/ppo/a2c/ddpg/ac")

    # trainer
    parser.add_argument('--buffer_capacity', default=int(10000), type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--clip_param', default=0.2, type=int)
    parser.add_argument('--max_grad_norm', default=0.5, type=int)
    parser.add_argument('--ppo_update_time', default=10, type=int)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--hidden_size', default=100)
    parser.add_argument('--max_episode', default=1000, type=int)
    parser.add_argument('--target_replace', default=100)
    parser.add_argument('--train_frequency', default=100)
    parser.add_argument('--tau', default=0.02)
    parser.add_argument('--update_freq', default=10)

    parser.add_argument('--reload_config', default=False, type=bool)
    parser.add_argument('--run_redo', default=None, type=int)

    # exploration
    parser.add_argument('--epsilon', default=0.02)  # cartpole 0.2 # mountaincar 1
    parser.add_argument('--epsilon_end', default=0.05)  # cartpole 0.05 # mountaincar 0.05

    # evaluation
    parser.add_argument('--evaluate_rate', default=50)

    # seed
    parser.add_argument('--seed_nn', default=1, type=int)
    parser.add_argument('--seed_np', default=1, type=int)
    parser.add_argument('--seed_random', default=1, type=int)

    args = parser.parse_args()

    return args
