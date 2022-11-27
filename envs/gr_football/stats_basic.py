import numpy as np


class StatsCaculator:
    def __init__(self):
        self.stats = None
        self.reset()

    def reset(self):
        self.player_last_hold_ball = -1
        self.cumulative_shot_reward = None
        self.passing_flag = [False] * 11
        self.shot_flag = [False] * 11
        self.steal_ball_recording = False
        self.lost_ball_recording = False

        self.stats = {
            "reward": 0,
            "win": 0,
            "lose": 0,
            "score": 0,
            "my_goal": 0,
            "goal_diff": 0,
            "num_pass": 0,
            "num_shot": 0,
            "total_pass": 0,
            "good_pass": 0,
            "bad_pass": 0,
            "total_shot": 0,
            "good_shot": 0,
            "bad_shot": 0,
            "total_possession": 0,
            "tackle": 0,
            "get_tackled": 0,
            "interception": 0,
            "get_intercepted": 0,
            "total_move": 0,
        }

    def calc_stats(self, state, reward, idx):
        obs = state.obs
        prev_obs = state.prev_obs
        action = state.action

        self.stats["reward"] += reward
        self.stats["num_pass"] += 1 if action in [9, 10, 11] else 0
        self.stats["num_shot"] += 1 if action == 12 else 0

        if idx == 0:
            # only count once
            my_score, opponent_score = obs["score"]
            if my_score > opponent_score:
                self.stats["win"] = 1
                self.stats["lose"] = 0
                self.stats["score"] = 1
            elif my_score < opponent_score:
                self.stats["win"] = 0
                self.stats["lose"] = 1
                self.stats["score"] = 0
            else:
                self.stats["win"] = 0
                self.stats["lose"] = 0
                self.stats["score"] = 0.5
            self.stats["my_goal"] = my_score
            self.stats["goal_diff"] = my_score - opponent_score

        self.count_possession(obs)
        self.count_pass(obs, action)
        self.count_shot(prev_obs, obs, action)
        self.count_getpossession(prev_obs, obs)
        self.count_losepossession(prev_obs, obs)
        self.count_move(prev_obs, obs)

    def count_possession(self, obs):
        if (
            obs["ball_owned_team"] == 0 and obs["active"] == 1
        ):  # compute only once for the whole team
            self.stats["total_possession"] += 1

    def count_pass(self, obs, player_action):

        for i, p in enumerate(self.passing_flag):
            if p:  # if passing
                if obs["ball_owned_team"] == 0 and obs["active"] == i:
                    pass
                else:
                    if obs["ball_owned_team"] == 0 and obs["ball_owned_player"] != i:
                        self.passing_flag[i] = False
                        self.stats["good_pass"] += 1
                    elif obs["ball_owned_team"] == -1:
                        pass
                    elif obs["ball_owned_team"] == 1 and obs["active"] == i:
                        self.stats["bad_pass"] += 1
                        self.passing_flag[i] = False

        if player_action == 9 or player_action == 10 or player_action == 11:
            if (
                obs["ball_owned_team"] == 0
                and not self.passing_flag[obs["active"]]
                and (obs["active"] == obs["ball_owned_player"])
            ):

                self.passing_flag[obs["active"]] = True
                self.stats["total_pass"] += 1

    def count_shot(self, prev_obs, obs, player_action):

        for i, p in enumerate(self.shot_flag):
            if p:
                if prev_obs["score"][0] < obs["score"][0] and obs["active"] == i:
                    self.stats["good_shot"] += 1
                    self.shot_flag[i] = False
                else:

                    if (
                        obs["ball_owned_team"] == 0
                        and obs["active"] == i
                        and obs["ball_owned_player"] == i
                    ):  # havnt left the player
                        pass
                    else:
                        if (
                            obs["ball_owned_team"] == 0
                            and obs["ball_owned_player"] != i
                            and obs["active"] == i
                        ):
                            self.stats["bad_shot"] += 1
                            self.shot_flag[i] = False

                        elif obs["ball_owned_team"] == -1:
                            pass
                        elif obs["ball_owned_team"] == 1:
                            self.stats["bad_shot"] += 1
                            self.shot_flag[i] = False
                        else:
                            pass

        if player_action == 12:
            if (
                obs["ball_owned_team"] == 0
                and not self.shot_flag[obs["active"]]
                and (obs["active"] == obs["ball_owned_player"])
            ):

                self.shot_flag[obs["active"]] = True
                self.stats["total_shot"] += 1

    def count_getpossession(self, prev_obs, obs):

        if prev_obs["score"][1] < obs["score"][1]:
            self.steal_ball_recording = (
                False  # change of ball ownership due to opponent's goal
            )
            return

        if (
            obs["game_mode"] == 3
        ):  # change of ball ownership from free kick, this is likely due to opponent offside
            self.steal_ball_recording = (
                False  # change of ball ownership due to opponent's goal
            )
            return

        if self.steal_ball_recording:
            if obs["ball_owned_team"] == -1:
                pass
            elif obs["ball_owned_team"] == 1:
                self.steal_ball_recording = False
            elif (
                obs["ball_owned_team"] == 0 and obs["ball_owned_player"] == 0
            ):  # our goalkeeper intercept the ball
                self.steal_ball_recording = False
            elif (
                obs["ball_owned_team"] == 0
                and obs["ball_owned_player"] != 0
                and obs["active"] == obs["ball_owned_player"]
            ):
                self.steal_ball_recording = False
                self.stats[
                    "interception"
                ] += 1  # only reward the agent stealing the ball (can we make it team reward?)

        if (
            prev_obs["ball_owned_team"] == 1 and prev_obs["ball_owned_player"] != 0
        ) and obs["ball_owned_team"] == 0:
            if obs["active"] == obs["ball_owned_player"]:
                self.stats["tackle"] += 1
        elif (
            prev_obs["ball_owned_team"] == 1 and prev_obs["ball_owned_player"] != 0
        ) and obs["ball_owned_team"] == -1:
            self.steal_ball_recording = True
        else:
            pass

    def count_losepossession(self, prev_obs, obs):

        if prev_obs["score"][0] < obs["score"][0]:
            self.lost_ball_recording = (
                False  # change of ball ownership due to ours goal
            )
            return

        if self.lost_ball_recording:
            if obs["ball_owned_team"] == -1:
                pass
            elif obs["ball_owned_team"] == 0:  # back to our team
                self.lost_ball_recording = False
                # can add reward here
            else:  # opponent own it
                if self.last_hold_player == 0:  # our goalkeeper lose the ball
                    self.lost_ball_recording = False

                if obs["active"] == self.last_hold_player:
                    self.lost_ball_recording = False
                    self.stats["get_intercepted"] += 1

        if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 1:
            if (
                obs["active"] == prev_obs["ball_owned_player"]
            ):  # current player is the last holding player
                self.stats["get_tackled"] += 1

        elif prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == -1:
            self.lost_ball_recording = True
            self.last_hold_player = prev_obs["ball_owned_player"]

    def count_move(self, prev_obs, obs):
        current_player = obs["active"]
        left_position_move = np.sum(
            (prev_obs["left_team"][current_player] - obs["left_team"][current_player])
            ** 2
        )
        self.stats["total_move"] += left_position_move
