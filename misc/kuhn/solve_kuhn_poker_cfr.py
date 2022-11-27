"""
https://colab.research.google.com/github/deepmind/open_spiel/blob/master/open_spiel/colabs/CFR_and_REINFORCE.ipynb"
"""
import numpy as np
import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib


class CFRSolver:
    def __init__(self):
        self.game = pyspiel.load_game("kuhn_poker")
        self.regret_sums = {}
        self.regrets = None
        self.action_probs = {}
        self.index2state = {}
        self.num_players = self.game.num_players()
        self.num_actions = self.game.num_distinct_actions()
        assert self.num_players == 2 and self.num_actions == 2

        self.max_steps = 129

    def index_state(self, player, state):
        return "".join(
            [str(int(num)) for num in state.information_state_tensor(player)]
        )

    def get_official_format_policy(self):
        policy = policy_lib.TabularPolicy(self.game)
        for state_index, action_probs in self.action_probs.items():
            state = self.index2state[state_index]
            new_state_index = policy.state_index(state)
            policy.action_probability_array[new_state_index] = action_probs
        return policy

    def train(self):
        self.eval_steps = []
        self.eval_nash_conv = []

        # we use simutaneously updating
        for step in range(self.max_steps):
            self.regrets = {}

            # compute immediate counterfactual regret
            self.step()

            # compute the new regret-matching policy
            floored_regrets = {k: np.maximum(v, 1e-16) for k, v in self.regrets.items()}
            curr_policy = {k: v / np.sum(v) for k, v in floored_regrets.items()}

            # print(curr_policy)

            # update the average policy
            lr = 1 / (1 + step)
            for k, v in self.action_probs.items():
                self.action_probs[k] = (1 - lr) * v + lr * curr_policy[k]

            if step & (step - 1) == 0:
                policy = self.get_official_format_policy()
                nash_conv = exploitability.nash_conv(self.game, policy)
                self.eval_steps.append(step)
                self.eval_nash_conv.append(nash_conv)
                print(f"Nash conv after step {step} is {nash_conv}")

    def step(self):
        state = self.game.new_initial_state()
        # including a chance node
        reaching_probs = np.ones(1 + self.num_players, dtype=np.float32)
        self.traverse(state, reaching_probs)

    def new_reaching_probs(self, reaching_probs, player, action_prob):
        new_reaching_probs = reaching_probs.copy()
        new_reaching_probs[player] *= action_prob
        return new_reaching_probs

    def get_action_probs(self, state_index, state, legal_actions):
        if state_index not in self.action_probs:
            self.index2state[state_index] = state
            probs = np.zeros(self.num_actions, dtype=np.float32)
            probs[np.array(legal_actions, dtype=int)] = 1
            self.action_probs[state_index] = probs / np.sum(probs)
        return self.action_probs[state_index]

    def traverse(self, state, reaching_probs):
        """

        return:
            state_values: definition in RL
        """
        if state.is_terminal():
            return state.returns()
        elif state.is_chance_node():
            return sum(
                prob
                * self.traverse(
                    state.child(action),
                    self.new_reaching_probs(reaching_probs, -1, prob),
                )
                for action, prob in state.chance_outcomes()
            )
        else:
            player = state.current_player()
            index = self.index_state(player, state)

            state_action_values = np.zeros(
                (self.num_actions, self.num_players), dtype=np.float32
            )
            legal_actions = state.legal_actions()
            action_probs = self.get_action_probs(index, state, legal_actions)
            for action in legal_actions:
                action_prob = action_probs[action]
                state_action_values[action] = self.traverse(
                    state.child(action),
                    self.new_reaching_probs(reaching_probs, player, action_prob),
                )

            # exclude his own reaching prob
            cfr_prob = np.prod(reaching_probs[:player]) * np.prod(
                reaching_probs[player + 1 :]
            )
            state_values = np.einsum("ij,i->j", state_action_values, action_probs)

            if index not in self.regrets:
                self.regrets[index] = np.zeros(self.num_actions, dtype=np.float32)

            legal_action_mask = np.zeros(self.num_actions, dtype=np.float32)
            legal_action_mask[np.array(legal_actions)] = 1.0
            self.regrets[index] += (
                cfr_prob
                * legal_action_mask
                * (state_action_values[:, player] - state_values[player])
            )

            return state_values


if __name__ == "__main__":
    solver = CFRSolver()
    solver.train()
