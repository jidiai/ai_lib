from runner import Runner
import argparse

if __name__ == '__main__':
    # set env and algo
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="gridworld", type=str)
    parser.add_argument('--algo', default="tabularq", type=str,
                        help="tabularq/sarsa/dqn/ppo/ddpg/ac/ddqn/duelingq/sac/pg/sac/td3")

    parser.add_argument('--reload_config', action='store_true')  # 加是true；不加为false
    args = parser.parse_args()

    print("================== args: ", args)
    print("== args.reload_config: ", args.reload_config)

    runner = Runner(args)
    runner.run()