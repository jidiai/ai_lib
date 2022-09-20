def default_table_name(agent_id,policy_id,share_policies):
    if not share_policies:
        return "{}_{}".format(agent_id,policy_id)
    else:
        return policy_id