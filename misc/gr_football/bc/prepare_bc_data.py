dataset_dir="/root/autodl-tmp/data/saltyfish_dataset_1"
output_path="/root/autodl-tmp/data/saltyfish_dataset_1_bc_data_basic.pkl"
num_workers=10

from model.gr_football.basic_11.encoder_basic import FeatureEncoder
from envs.gr_football.state import State
from utils.logger import Logger

import os
import pickle as pkl
import json
import numpy as np

from utils.episode import EpisodeKey
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

def convert(args):
    ctr,total,data_path=args
    feature_encoder=FeatureEncoder()
    with open(data_path) as f:
        data=json.load(f)
    info=data["info"]
    try:
        team_idx=info["TeamNames"].index("SaltyFish")
    except ValueError:
        print("SaltyFish not in {}".format(data_path))
        return []

    observations=[]
    actions=[]
    state=State()
    for idx,step in enumerate(data["steps"]):
        action=step[team_idx]["action"]
        if idx!=0:
            actions.append(action)
            state.update_action(action[0])
        raw_observation=step[team_idx]["observation"]["players_raw"][0]
        state.update_obs(raw_observation)
        if idx!=len(data["steps"])-1:
            observation=feature_encoder.encode([state])
            observations.append(observation)
    samples=[{EpisodeKey.EXPERT_OBS:np.array([observation]),EpisodeKey.EXPERT_ACTION: np.array([action])} for observation,action in zip(observations,actions)]
    Logger.info("Converted {}/{}".format(ctr+1,total))
    return samples

file_names=os.listdir(dataset_dir)
data_paths=[(idx,len(file_names),os.path.join(dataset_dir,filename)) for idx,filename in enumerate(file_names)]
with Pool(num_workers) as executor:
    results=executor.map(convert,data_paths)

samples_all=[]
for idx,samples in enumerate(results):
    samples_all.extend(samples)
    
with open(output_path,"wb") as f:
    pkl.dump(samples_all,f)
