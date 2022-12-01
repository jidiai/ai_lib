import_commmand = "from .snakes import *\nfrom .reversi import *\nfrom .gobang import *\nfrom .sokoban import *\nfrom .ccgame import *\nfrom .football import *\nfrom .MiniWorld import *\nfrom .minigrid import *\nfrom .particleenv import *\nfrom .overcookedai import *\nfrom .magent import *\nfrom .gridworld import *\nfrom .cliffwalking import *\nfrom .smarts_jidi import *\nfrom .sc2 import *\nfrom .olympics_running import *\nfrom .smarts_ngsim import *\nfrom .gym_robotics import *\nfrom .chessandcard import *\nfrom .chinesechess import *\nfrom .logisticsenv import *\n"


for i in import_commmand.split('\n'):
    try:
        exec(i)
    except:
        print(f'FAIL: {i}')

