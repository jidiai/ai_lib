
<img src="imgs/Jidi%20logo.png" width='300px'>


# JidiRLlib: A general reinforcement learning library
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Release Version](https://img.shields.io/badge/release-2.0-red.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

**JidiRLlib is a general distributed-RL library that integrates  Single-Agent RL, Multi-Agent RL and Population-based 
Training framework (i.e. Self-Play, PSRO, League Training). It inherits its structure 
from [DB-Football](https://github.com/Shanghai-Digital-Brain-Laboratory/DB-Football)** and **[MALib](https://github.com/sjtu-marl/malib)**.


## Quick Start

You can install Jidi AiLib on your own personal computer or workstation. To install it, please follow the instruction below.

```bash
git clone https://github.com/YanSong97/ailib_v2.0.git
cd ailib_v2
```

your can use virtual environment:
```bash
python3 -m venv ailib-venv
source ailib-venv/bin/activate
python3 -m pip install -e .
```

## Navigation

<img src='https://github.com/jidiai/ai_lib/blob/V2/imgs/Jidi%20Ailib.svg'>


## Contribution
若想在框架基础上添加环境或算法，可跟随下列步骤：
1. 在`./envs/`文件夹内添加环境文件，环境类型可为**单边单智能体**环境如`gym`，**双边多智能体**环境如`gr_football`,**顺序决策**环境如`poker`
(一次只有一方而不是双方同时决策)。可根据所提供的环境例子接入你想要的环境，不同环境所使用的`rollout_fn`不一样。
2. 在选定环境类型后，需选定`rollout_fn`类型，在`./rollout/`文件夹内，`rollout_func.py`适用于双边多智能体环境(e.g. gr_football),
``rollout_func_aec.py``适用于顺序决策环境(e.g.poker), `rollout_func_seg.py`适用于单边单智能体环境(e.g. gym).
3. 选定好后在`./expr/`文件夹内写好config
4. run ``main_pbt.py`` or `main_marl.py`




## Supported Learing Paradims
- [x] **BC**: serves for behavior cloning
- [x] **SARL**: serves for single-agent reinforcement learning
- [x] **MARL**: serves for multi-agent reinforcement learning
- [x] **Self-play**: general empirical game theory learning
- [x] **PSRO**: policy-space response oracle
- [ ] **League Training**

## Supported Training environments
- [x] **Discrete Gym Envs**
- [x] **Continuous Gym Envs**
- [x] **Pettingzoo Envs**
- [x] **GRFootball**
- [x] **Open-spiel Kuhn Poker**
- [x] **Open-spiel Leduc Poker**
- [x] **Connect-Four**
- [x] **MPE**

## Supported Baseline Algorithms
- [x] **(MA)PPO**
- [x] **table-q learning**
- [x] **(MA)DQN**
- [x] **DDPG**
- [x] **SAC**
- [ ] **QMIX**

