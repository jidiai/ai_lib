# import_commmand="from .snakes import *\nfrom .reversi import *\nfrom .gobang import *\nfrom .sokoban import *\nfrom .ccgame import *\nfrom .football import *\nfrom .MiniWorld import *\nfrom .minigrid import *\nfrom .overcookedai import *\nfrom .magent import *\nfrom .gridworld import *\nfrom .cliffwalking import *\nfrom .smarts_jidi import *\nfrom .sc2 import *\nfrom .smarts_ngsim import *\nfrom .gym_robotics import *\nfrom .chessandcard import *\nfrom .chinesechess import *\nfrom .logisticsenv import *\nfrom .olympics_tablehockey import *\nfrom .olympics_football import *\nfrom .olympics_wrestling import *\nfrom .olympics_billiard import *\nfrom .olympics_running import *\nfrom .mpe_jidi import *\nfrom .olympics_curling import *\nfrom .delivery import *\nfrom .logisticsenv2 import *\nfrom .olympics_integrated import *\nfrom .wilderness import *\nfrom .revive_refrigerator import *\nfrom .finrl_stocktrading import *\nfrom .olympics_billiard_competition import *"

import_commmand = [
"from .snakes import *",
"from .reversi import *",
"from .gobang import *",
"from .sokoban import *",
"from .ccgame import *",
"from .football import *",
"from .MiniWorld import *",
"from .minigrid import *",
"from .overcookedai import *",
"from .magent import *",
"from .gridworld import *",
"from .cliffwalking import *",
"from .smarts_jidi import *",
"from .sc2 import *",
"from .smarts_ngsim import *",
"from .gym_robotics import *",
"from .chessandcard import *",
"from .chinesechess import *",
"from .logisticsenv import *",
"from .olympics_tablehockey import *",
"from .olympics_football import *",
"from .olympics_wrestling import *",
"from .olympics_billiard import *",
"from .olympics_running import *",
"from .mpe_jidi import *",
"from .olympics_curling import *",
"from .delivery import *",
"from .logisticsenv2 import *",
"from .olympics_integrated import *",
"from .wilderness import *",
"from .revive_refrigerator import *",
"from .finrl_stocktrading import *",
"from .olympics_billiard_competition import *",
"from .fourplayers_nolimit_texas_holdem import *",
"from .bridge import *",
"from .overcookedai_integrated import *",
"from .taxing_gov import *",
"from .taxing_household import *",
]

for i in import_commmand:
    try:
        exec(i)
    except:
        print(f'FAIL: {i}')
