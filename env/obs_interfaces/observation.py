# -*- coding:utf-8  -*-
obs_type = ["grid", "vector", "dict"]


class GridObservation(object):
    def get_grid_observation(self, current_state, player_id, info_before):
        raise NotImplementedError

    def get_grid_many_observation(self, current_state, player_id_list, info_before=''):
        all_obs = []
        for i in player_id_list:
            all_obs.append(self.get_grid_observation(current_state, i, info_before))
        return all_obs


class VectorObservation(object):
    def get_vector_observation(self, current_state, player_id, info_before):
        raise NotImplementedError

    def get_vector_many_observation(self, current_state, player_id_list, info_before=''):
        all_obs = []
        for i in player_id_list:
            all_obs.append(self.get_vector_observation(current_state, i, info_before))
        return all_obs


class DictObservation(object):
    def get_dict_observation(self, current_state, player_id, info_before):
        raise NotImplementedError

    def get_dict_many_observation(self, current_state, player_id_list, info_before=''):
        all_obs = []
        for i in player_id_list:
            all_obs.append(self.get_dict_observation(current_state, i, info_before))
        return all_obs