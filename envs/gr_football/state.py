from .tools import geometry as g
import numpy as np


class State:
    def __init__(self):
        self.obs_list = []
        self.action_list = []
        self.last_ball_owned_team = None
        self.last_ball_owned_player = None
        self.last_loffside = np.zeros(5, np.float32)
        self.last_roffside = np.zeros(5, np.float32)
        self.first_kicker = None

    def update_obs(self, obs):
        self.obs_list.append(obs)
        self.update_last_ball_owned()

    def update_action(self, action):
        self.action_list.append(action)

    @property
    def obs(self):
        return self.obs_list[-1] if len(self.obs_list) >= 1 else None

    @property
    def prev_obs(self):
        return self.obs_list[-2] if len(self.obs_list) >= 2 else None

    @property
    def action(self):
        return self.action_list[-1] if len(self.action_list) >= 1 else None

    @property
    def prev_action(self):
        return self.action_list[-2] if len(self.action_list) >= 2 else None

    def update_last_ball_owned(self):
        last_ball_owned_team, last_ball_owned_player = self.get_last_ball_owned(
            self.obs, self.prev_obs
        )
        if last_ball_owned_team is not None:
            self.last_ball_owned_team = last_ball_owned_team
            self.last_ball_owned_player = last_ball_owned_player
            if self.first_kicker is None:
                self.first_kicker = 1 if self.last_ball_owned_team == 0 else 0

        assert self.last_ball_owned_team != -1 and self.last_ball_owned_player != -1

    def get_last_ball_owned(self, obs, prev_obs=None):
        if prev_obs is None:
            team, idx, dist = self.get_closest_player_to_ball(obs)
            owned_team = 1 if team == "right" else 0
            return owned_team, idx
        if obs["game_mode"] == 0:  # Normal
            if obs["ball_owned_team"] != -1:
                return obs["ball_owned_team"], obs["ball_owned_player"]
            elif obs["ball_owned_team"] == -1 and self.scored(prev_obs, obs):
                # another team kick off now
                team = "right" if self.scored(prev_obs, obs) == 1 else "left"
                team, idx, dist = self.get_closest_player_to_ball(obs, team)
                owned_team = 1 if team == "right" else 0
                return owned_team, idx
            else:
                # TODO jh need code review
                # NOTE direct passing: this is hard, not sure it is a correct implementation
                team, idx, dist = self.get_closest_player_to_ball(obs)
                prev_ball_coord_speed = prev_obs["ball_direction"]
                ball_coord_speed = obs["ball_direction"]
                speed_change = g.get_speed(ball_coord_speed - prev_ball_coord_speed)
                if (
                    dist < g.BALL_CONTROLLED_DIST * 1.5
                    and g.tz(obs["ball"][-2]) < g.BALL_CONTROLLED_HEIGHT
                    and speed_change > g.BALL_SPEED_VARIATION_THRESH
                ):
                    owned_team = 1 if team == "right" else 0
                    return owned_team, idx
                else:
                    return None, None
        else:  # KickOff,GoalKick,FreeKick,ThrowIn,Penalty
            # just see who is closest to the ball.
            team, idx, _ = self.get_closest_player_to_ball(obs)
            owned_team = 1 if team == "right" else 0
            return owned_team, idx

    def _get_closest_player_to_ball(self, obs, team):
        ball = obs["ball"]
        players = obs["{}_team".format(team)]
        dists = g.get_dist(ball[:2], players)
        idx = np.argmin(dists)
        return idx, dists[idx]

    def get_closest_player_to_ball(self, obs, team=None):
        if team is not None:
            idx, dist = self._get_closest_player_to_ball(obs, team)
            return team, idx, dist
        l_idx, l_dist = self._get_closest_player_to_ball(obs, "left")
        r_idx, r_dist = self._get_closest_player_to_ball(obs, "right")
        if l_dist < r_dist:
            return "left", l_idx, l_dist
        else:
            return "right", r_idx, r_dist

    def scored(self, prev_obs, obs):
        if prev_obs is None:
            return 0
        if prev_obs["score"][0] < obs["score"][0]:
            return 1
        elif prev_obs["score"][1] < obs["score"][1]:
            return -1
        return 0

    def get_offside(self, obs):
        """
        Copied from wekick.
        """
        ball = np.array(obs["ball"][:2])
        ally = np.array(obs["left_team"])
        enemy = np.array(obs["right_team"])

        if obs["game_mode"] != 0:
            self.last_loffside = np.zeros(5, np.float32)
            self.last_roffside = np.zeros(5, np.float32)
            return np.zeros(5, np.float32), np.zeros(5, np.float32)

        need_recalc = False
        effective_ownball_team = -1
        effective_ownball_player = -1

        if obs["ball_owned_team"] > -1:
            effective_ownball_team = obs["ball_owned_team"]
            effective_ownball_player = obs["ball_owned_player"]
            need_recalc = True
        else:
            ally_dist = np.linalg.norm(ball - ally, axis=-1)
            enemy_dist = np.linalg.norm(ball - enemy, axis=-1)
            if np.min(ally_dist) < np.min(enemy_dist):
                if np.min(ally_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 0
                    effective_ownball_player = np.argmin(ally_dist)
            elif np.min(enemy_dist) < np.min(ally_dist):
                if np.min(enemy_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 1
                    effective_ownball_player = np.argmin(enemy_dist)

        if not need_recalc:
            return self.last_loffside, self.last_roffside

        left_offside = np.zeros(5, np.float32)
        right_offside = np.zeros(5, np.float32)

        if effective_ownball_team == 0:
            right_xs = [obs["right_team"][k][0] for k in range(0, 5)]
            right_xs = np.array(right_xs)
            right_xs.sort()

            offside_line = max(right_xs[-2], ball[0])

            for k in range(1, 5):
                if (
                    obs["left_team"][k][0] > offside_line
                    and k != effective_ownball_player
                    and obs["left_team"][k][0] > 0.0
                ):
                    left_offside[k] = 1.0
        else:
            left_xs = [obs["left_team"][k][0] for k in range(0, 5)]
            left_xs = np.array(left_xs)
            left_xs.sort()

            offside_line = min(left_xs[1], ball[0])

            for k in range(1, 5):
                if (
                    obs["right_team"][k][0] < offside_line
                    and k != effective_ownball_player
                    and obs["right_team"][k][0] < 0.0
                ):
                    right_offside[k] = 1.0

        self.last_loffside = left_offside
        self.last_roffside = right_offside

        return left_offside, right_offside
