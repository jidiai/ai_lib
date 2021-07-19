import os

def ini_agents(args):

    file_path = os.path.dirname(os.path.abspath(__file__)) + "/algo/" + args.algo + "/" + args.algo
    print(file_path)
    import_path = '.'.join(file_path.split('/')[-3:])
    algo_name = args.algo.upper()
    import_s = "from %s import %s as agent" % (import_path, algo_name)
    print(import_s)
    exec(import_s, globals())

    return agent(args)

