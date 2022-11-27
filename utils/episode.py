class EpisodeKey:
    """Unlimited buffer"""

    CUR_OBS = "observation"
    NEXT_OBS = "next_observation"
    ACTION = "action"
    ACTION_MASK = "action_mask"
    NEXT_ACTION_MASK = "next_action_mask"
    REWARD = "reward"
    DONE = "done"
    # XXX(ziyu): Change to 'logits' for numerical issues.
    ACTION_DIST = "action_logits"
    ACTION_PROB = "action_prob"
    ACTION_PROBS = "action_probs"
    # XXX(ming): seems useless
    INFO = "infos"

    ACTIVE_MASK = "active_mask"

    # optional
    STATE_VALUE = "state_value_estimation"
    STATE_ACTION_VALUE = "state_action_value_estimation"
    CUR_STATE = "state"  # current global state
    NEXT_STATE = "next_state"  # next global state
    LAST_REWARD = "last_reward"
    RETURN = "return"

    # post process
    ACC_REWARD = "accumulate_reward"
    ADVANTAGE = "advantage"
    STATE_VALUE_TARGET = "state_value_target"

    # model states
    RNN_STATE = "rnn_state"
    ACTOR_RNN_STATE = "ACTOR_RNN_STATE"
    CRITIC_RNN_STATE = "CRITIC_RNN_STATE"

    # expert
    EXPERT_OBS = "expert_obs"
    EXPERT_ACTION = "expert_action"
