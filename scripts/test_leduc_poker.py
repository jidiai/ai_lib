import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib
import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

game=pyspiel.load_game("leduc_poker")

print(game.get_type().pretty_print())

policy = policy_lib.TabularPolicy(game)
# print(policy.states_per_player)
# print(policy.action_probability_array)
print(exploitability.nash_conv(game, policy))

initial_state = game.new_initial_state()

state=initial_state

def sample(actions_and_probs):
    actions, probs = zip(*actions_and_probs)
    return np.random.choice(actions, p=probs)

while state.is_chance_node():
    p = state.chance_outcomes()
    a=sample(p)
    state.apply_action(a)

idx=policy.state_index(state)
print(state)
print("here",state.observation_tensor(0))

# print(state)
# print(policy.state_index(state))

# from open_spiel.python.rl_environment import TimeStep,Environment

# env=Environment(
#     game="leduc_poker",
#     discount=1.0,
#     players=2
# )

# data=env.reset()

# print(data)

# state=data.observations["info_state"][0]
# print(state)
