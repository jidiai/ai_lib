from EnvWrapper.BaseWrapper import BaseWrapper
from pathlib import Path
import sys
import os
import numpy as np

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
print("+",base_dir)
from env.chooseenv import make
env = make("MiniWorld-OneRoom-v0")

import cv2


def state_processed(observation):
    observation = np.array(observation,dtype=np.uint8).reshape(60,80,3)
    img = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    gs_img = cv2.GaussianBlur(img,(3,3),0)
    hsv_img = cv2.cvtColor(gs_img, cv2.COLOR_BGR2HSV)
    erode_hsv = cv2.erode(hsv_img, None, iterations=2)
    inRange_hsv = cv2.inRange(erode_hsv, np.array([0,60,60]), np.array([6,255,255]))
    cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not cnts:
        return [-10, -10]
    c = max(cnts, key = cv2.contourArea)
    M=cv2.moments(c)
    cx = M['m10']/(M['m00']+1) # 求x坐标
    cy = M['m01']/(M['m00']+1) # 求y坐标
    img=cv2.circle(img ,(int(cx),int(cy)),2,(0,0,255),4) #画出重心
    return [cx,cy]

class MiniWorld_OneRoom_v0(BaseWrapper):
    def __init__(self):
        self.env = env
        super().__init__(self.env)

    def get_actionspace(self):
        print("##", self.env.action_dim)
        return self.env.action_dim

    def get_observationspace(self):
        print("##", self.env.input_dimension.shape)
        return self.env.input_dimension.shape

    def step(self, action, train=True):
        #action = action_wrapper([action])
        next_state, reward, done, _, _ = self.env.step(action)
        reward  =np.array(reward,dtype=np.float)
        reward *= 100
        if not done:
            reward -= 0.5
        next_state = np.array(next_state[0]['obs']).reshape(60,80,3)
        return [{'obs':next_state,"controlled_player_index": 0}], reward, done, _, _
    
    def reset(self):
        state = self.env.reset()
        state = np.array(state[0]['obs']).reshape(60,80,3)
        return [{"obs": state, "controlled_player_index": 0}]

    def close(self):
        pass

    def set_seed(self, seed):
        self.env.set_seed(seed)
'''
def action_wrapper(joint_action):
    joint_action_ = []
    for a in range(env.n_player):
        action_a = joint_action[a]["action"]
        each = [0] * env.action_dim
        each[action_a] = 1
        action_one_hot = [[each]]
        joint_action_.append([action_one_hot[0][0]])
    return joint_action_
'''