import copy
import itertools
import os
from pathlib import Path

import pandas as pd
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TEST_START_DATE,
    TEST_END_DATE,
)
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

from env.simulators.game import Game
from utils.box import Box


current_dir = str(Path(__file__).resolve().parent)
TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2021-10-31'


class FinRL_StockTradingEnv(Game):

    def __init__(self, conf):
        super(FinRL_StockTradingEnv, self).__init__(conf['n_player'], conf['is_obs_continuous'],
                                                    conf['is_act_continuous'], conf['game_name'], conf['agent_nums'],
                                                    conf['obs_type'])
        data_file = os.path.join(current_dir, "finrl", "processed_stock_price.csv")
        # df = pd.read_csv(data_file)
        # df.sort_values(['date', 'tic'], ignore_index=True)
        # fe = FeatureEngineer(
        #     use_technical_indicator=True,
        #     tech_indicator_list=INDICATORS,
        #     use_vix=True,
        #     use_turbulence=True,
        #     user_defined_feature=False)
        #
        processed = pd.read_csv(data_file)

        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
        combination = list(itertools.product(list_date, list_ticker))

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"],
                                                                                  how="left")
        processed_full = processed_full[processed_full['date'].isin(processed['date'])]
        processed_full = processed_full.sort_values(['date', 'tic'])

        processed_full = processed_full.fillna(0)
        train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
        stock_dimension = len(train.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
        buy_cost_list = sell_cost_list = [0.001] * stock_dimension
        num_stock_shares = [0] * stock_dimension

        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }

        e_train_gym = StockTradingEnv(df=train, **env_kwargs)
        self.env_core, _ = e_train_gym.get_sb_env()
        self.joint_action_space = self.set_action_space()

        self.step_cnt = 0
        self.n_return = [0] * self.n_player

        self.current_state = self.env_core.reset()
        self.all_observes = self.get_all_observes()
        self.won = {}
        self.done = [False]
        self.init_info = None

    def set_action_space(self):
        return [[Box(low=-1, high=1, shape=(1,)) for _ in range(self.env_core.action_space.shape[0])]]

    def reset(self):
        self.step_cnt = 0
        self.n_return = [0] * self.n_player

        self.current_state = self.env_core.reset()
        self.all_observes = self.get_all_observes()
        self.won = {}
        self.done = False

    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            each = {"obs": copy.deepcopy(self.current_state), "controlled_player_index": i}
            all_observes.append(each)

        return all_observes

    def step(self, joint_action):
        self.step_cnt += 1
        decoded_action = self.decode(joint_action)
        obs, reward, self.done, info = self.env_core.step(decoded_action)
        self.set_n_return(reward[0])
        self.current_state = obs
        all_observes = self.get_all_observes()
        # print("all_observes: ", all_observes[0]['obs'].shape)
        return all_observes, reward[0], self.done, info, ''

    def set_n_return(self, reward):
        for i in range(self.n_player):
            self.n_return[i] += reward

    def is_terminal(self):
        return self.done[0]

    def decode(self, joint_action):
        joint_action_decoded = []
        for _, action in enumerate(joint_action):
            joint_action_decoded.append(action[0])
        return joint_action_decoded

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def check_win(self):
        return ''

