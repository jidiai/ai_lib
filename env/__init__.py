import_commmand="from .snakes import *\nfrom .reversi import *\nfrom .gobang import *\nfrom .sokoban import *\nfrom .ccgame import *\nfrom .football import *\nfrom .MiniWorld import *\nfrom .minigrid import *\nfrom .overcookedai import *\nfrom .magent import *\nfrom .gridworld import *\nfrom .cliffwalking import *\nfrom .smarts_jidi import *\nfrom .sc2 import *\nfrom .smarts_ngsim import *\nfrom .gym_robotics import *\nfrom .chessandcard import *\nfrom .chinesechess import *\nfrom .logisticsenv import *\nfrom .olympics_tablehockey import *\nfrom .olympics_football import *\nfrom .olympics_wrestling import *\nfrom .olympics_billiard import *\nfrom .olympics_running import *\nfrom .mpe_jidi import *\nfrom .olympics_curling import *\nfrom .delivery import *\nfrom .logisticsenv2 import *\nfrom .olympics_integrated import *\nfrom .wilderness import *\nfrom .revive_refrigerator import *\nfrom .finrl_stocktrading import *"


for i in import_commmand.split('\n'):
    try:
        exec(i)
    except:
        print(f'FAIL: {i}')
