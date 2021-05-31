from gym_minigrid.wrappers import *
from env.simulators.gridgame import GridGame
import random
from env.obs_interfaces.observation import *
from utils.discrete import Discrete
import tkinter
import time
import gym
import gym_minigrid

# env = gym.make('MiniGrid-DoorKey-8x8-v0')

class MiniGrid(GridGame, GridObservation):
    def __init__(self, conf):
        colors = conf.get('colors', [(255, 255, 255), (0, 0, 0), (245, 245, 245)])
        super(MiniGrid, self).__init__(conf, colors)
        # self.renderer = Renderer()
        self.env_core = gym.make(conf['game_name'])
        self.action_dim = self.env_core.action_space.n
        self.input_dimension = self.env_core.observation_space['image'].shape
        # self.obs_type = [str(i) for i in str(conf["obs_type"]).split(',')]
        self.current_state = self.reset()
        self.is_act_continuous = False
        self.is_obs_continuous = True

    def step(self, joint_action):
        # action = self.decode(joint_action)
        # self.renderer.render(self._env_core.grid, self._env_core.agent_pos)
        action = joint_action
        info_before = self.step_before_info()
        next_state, reward, self.done, info_after = self.get_next_state(action)
        self.current_state = next_state
        if isinstance(reward, np.ndarray):
            reward = reward.tolist()
        reward = self.get_reward(reward)
        self.step_cnt += 1
        done = self.is_terminal()
        return next_state, reward, done, info_before, info_after

    def reset(self):
        obs = self.env_core.reset()['image']
        self.step_cnt = 0
        self.done = False
        self.current_state = obs
        return self.current_state

    def get_next_state(self, action):
        action = int(np.array(action[0][0]).argmax())
        observation, reward, done, info = self.env_core.step(action)
        return observation, reward, done, info

    def set_action_space(self):
        action_space = [[Discrete(7)] for _ in range(self.n_player)]
        return action_space

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        return self.done

    def get_grid_observation(self, current_state, player_id, info_before):
        return current_state

    def get_reward(self, reward):
        return [reward]

    def check_win(self):
        return True



class Renderer:
    def __init__(self):
        self.root = None
        self.color = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100])
}

    def _close_view(self):
        if self.root:
            self.root.destory()
            self.root = None
            self.canvas = None
        # self.done = True

    def render(self, map, agent_pos):
        time.sleep(0.1)
        scale = 30
        width = map.width * scale
        height = map.height * scale
        if self.root is None:
            self.root = tkinter.Tk()
            self.root.title("gym_minigrid")
            self.root.protocol("WM_DELETE_WINDOW", self._close_view)
            self.canvas = tkinter.Canvas(self.root, width=width, height=height)
            self.canvas.pack()

        self.canvas.delete(tkinter.ALL)
        self.canvas.create_rectangle(0, 0, width, height, fill="black")

        def fill_cell(x, y, color):
            self.canvas.create_rectangle(
                x * scale,
                y * scale,
                (x + 1) * scale,
                (y + 1) * scale,
                fill=color
            )

        for x in range(map.width):
            for y in range(map.height):
                if map.grid[int(x * width / scale + y)] != None:
                    fill_cell(x, y, map.grid[int(x * width / scale) + y].color)
                    # fill_cell(x,y,map[x,y])
        fill_cell(agent_pos[0], agent_pos[1], "Pink")

        self.root.update()