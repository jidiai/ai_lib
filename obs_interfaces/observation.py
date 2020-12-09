# -*- coding:utf-8  -*-
obs_type = ["grid"]


class GridObservation(object):
    def get_grid_observation(self, current_state, player_id):
        raise NotImplementedError

    def get_grid_many_observation(self, current_state, player_id_list):
        all_obs = []
        for i in player_id_list:
            all_obs.append(self.get_grid_observation(current_state, i))
        return all_obs



