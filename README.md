
<img src="imgs/Jidi%20logo.png" width='300px'>


# Jidi AiLib V2: A general reinforcement learning library
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Release Version](https://img.shields.io/badge/release-2.0-red.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

Jidi AiLib is a library that integrates many (MA)RL-related benchmarking environments and baselines to support various training and evaluation demands. 

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


## Supported Learing Paradims
- [x] **BC**: serves for behavior cloning
- [x] **SARL**: serves for single-agent reinforcement learning
- [x] **MARL**: serves for multi-agent reinforcement learning
- [x] **Self-play**: general empirical game theory learning
- [x] **PSRO**: policy-space response oracle
- [ ] **League Training**

## Supported Training environments
- [x] **Discrete Gym Envs**
- [ ] **Continuous Gym Envs**
- [ ] **Pettingzoo Envs**
- [x] **GRFootball**
- [x] **Open-spiel Kuhn Poker**
- [x] **Open-spiel Leduc Poker**
- [x] **Connect-Four**


## Supported Baseline Algorithms
- [x] **PPO**
- [x] **table-q learning**
- [x] **DQN**
- [ ] **DDPG**
- [ ] **SAC**

## Supported Benchmarking
- [x] **table-q learning in Kuhn Poker**
- [x] **table-q learning in Leduc Poker**
- [x] **PPO in GRFootball with Bot opponent**




# TODO
1. selective installation on environmental dependencies
2. add baseline envs
3. add benchmark environmens
4. ~~add SARL framework~~
5. ~~add PSRO, MARL benchmark graphs~~
6. **specify transformation from trained policy to ready-for-submission policy**
