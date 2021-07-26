import importlib


def ini_agents(args):
    agent_file_name = str("algo." + str(args.algo) + "." + str(args.algo))
    agent_file_import = importlib.import_module(agent_file_name)
    agent_class_name = args.algo.upper()
    # 实例化agent
    agent = getattr(agent_file_import, agent_class_name)(args)
    print("=========== agent: ",  agent)
    return agent

