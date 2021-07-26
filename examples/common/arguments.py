import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # set env and algo
    parser.add_argument('--scenario', default="classic_MountainCar-v0", type=str)
    parser.add_argument('--max_episodes', default=1000, type=int)
    parser.add_argument('--algo', default="dqn", type=str, help="dqn/ppo/a2c/ddpg/ac/ddqn/duelingq/sac")

    # trainer
    parser.add_argument('--buffer_capacity', default=int(256), type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.005, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--clip_param', default=0.2, type=float)
    parser.add_argument('--max_grad_norm', default=0.5, type=float)
    parser.add_argument('--ppo_update_time', default=10, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--hidden_size', default=100)
    parser.add_argument('--target_replace', default=100)
    parser.add_argument('--train_frequency', default=100)
    parser.add_argument('--tau', default=0.02)
    parser.add_argument('--update_freq', default=10)

    parser.add_argument('--learn_terminal', action='store_true') # 加是true；不加为false
    parser.add_argument('--learn_freq', default=1, type=int)

    parser.add_argument('--reload_config', action='store_true') # 加是true；不加为false
    parser.add_argument('--run_redo', default=None, type=int)

    parser.add_argument('--is_matrix', action='store_true') # 加是true；不加是false
    
    #added for sac
    parser.add_argument('--alpha_lr', default = 0.0001,type = float)
    parser.add_argument('--alpha', default = 0.2, type = float)
    parser.add_argument('--tune_entropy', default = False, type = bool)
    parser.add_argument('--target_entropy_ratio', default = 0.9, type = float)

    # exploration
    parser.add_argument('--epsilon', default=0.5)  # cartpole 0.2 # mountaincar 1
    parser.add_argument('--epsilon_end', default=0.05)  # cartpole 0.05 # mountaincar 0.05

    # evaluation
    parser.add_argument('--evaluate_rate', default=50)

    # seed
    parser.add_argument('--seed_nn', default=1, type=int)
    parser.add_argument('--seed_np', default=1, type=int)
    parser.add_argument('--seed_random', default=1, type=int)

    args = parser.parse_args()

    return args