
<img src="imgs/Jidi%20logo.png" width='300px'>

# **Jidi (及第) AiLib**            


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Release Version](https://img.shields.io/badge/release-1.0-red.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

This repo provides a reinforcement learning library designed for our online submission evaluation platform [Jidi(及第)](http://www.jidiai.cn/) which aims to offer fair benchmarking over various RL environments and host AI competitions on problems worth exploring.

**Version 2.0 (branch [V2](https://github.com/jidiai/ai_lib/tree/V2))**
New version of Jidi AiLib which supports Population-Based Training! Still under construction and stay tuned. PRs are welcome!

**Version 1.0 (branch master)**:
In the version V1.0, the repo contains the code of all the benchmark RL environment we included on [Jidi(及第)](http://www.jidiai.cn/) and simple-to-use training examples covering some classic baseline RL algorithms.


## **平台地址 (Our Website)**
[jidi_ai](http://www.jidiai.cn/)

## **快速开始 (Quick Start)**
You can install Jidi AiLib on your own personal computer or workstation. To install it, please follow the instruction below.

**Clone the repo**
```bash
git clone https://github.com/jidiai/ai_lib.git
cd ailib
```
**Build virtual environment**
```bash
python3 -m venv ailib-venv
source ailib-venv/bin/activate
python3 -m pip install -e .
```
or 
```bash
conda create -n ailib-venv python==3.7.5
conda activate ailib-venv
```
**Install necessary dependencies**
```bash
pip install -r requirement.txt
```

**Now have a go**
```bash
python examples/main.py --scenario classic_CartPole-v0 --algo dqn --reload_config 
```

## **训练例子 (Training Examples)**
We provide implementations and tuned training configurations of vairous baseline reinforcement learning algorithms. More details can be found in [./examples/](examples/README.md). Feel free to try it yourself.

We currently support the following benchmarking experiments:
<table>
    <tr>
        <td>Algo</td>
        <td>CartPole-v0</td> 
        <td>MountainCar-v0</td> 
        <td>Pendulum-v0</td> 
        <td>gridworld</td> 
   </tr>

[comment]: <> (   <tr>)

[comment]: <> (        <td colspan="2">合并行</td>    )

[comment]: <> (   </tr>)
   <tr>
        <td>RANDOM</td> 
        <td>√</td>
        <td>√</td> 
        <td>√</td> 
        <td>√</td> 
   </tr>
    <tr>
        <td>Q-learning</td> 
        <td>-</td> 
        <td>-</td> 
        <td> - </td> 
        <td> √ </td> 
    <tr>
        <td>Sarsa</td> 
        <td>-</td> 
        <td>-</td> 
        <td> - </td> 
        <td> √ </td> 
    <tr>
        <td>DQN</td> 
        <td>√</td> 
        <td>√</td> 
        <td> - </td> 
        <td> - </td> 
    <tr>
        <td>DDQN</td> 
        <td>√</td> 
        <td>√</td> 
        <td> - </td> 
        <td> - </td> 
    <tr>
        <td>Duelingq</td> 
        <td>√</td> 
        <td>√</td> 
        <td> - </td>
        <td> - </td> 
    <tr>
        <td>SAC</td>
        <td>√</td> 
        <td> √ </td> 
        <td> √ </td>
        <td> - </td> 

[comment]: <> (        <td rowspan="8">classic_CartPole-v0</td>)

   </tr>
    <tr>
        <td>PPO</td>
        <td>√</td> 
        <td> - </td> 
        <td> - </td>
        <td> - </td> 
   </tr>
    <tr>
        <td>PG</td>
        <td>√</td> 
        <td> - </td> 
        <td> - </td>
        <td> - </td> 
   </tr>
     <tr>
        <td>AC</td>
        <td>√</td> 
        <td> - </td> 
        <td> - </td> 
        <td> - </td> 
   </tr>
   </tr>
     <tr>
        <td>DDPG</td>
        <td>√</td> 
        <td> - </td> 
        <td> - </td>
        <td> - </td> 
     <tr>
        <td>TD3</td>
        <td> - </td> 
        <td> - </td> 
        <td> √ </td>
        <td> - </td> 
   </tr>
    

</table>




## **额外项目依赖 (Extra Dependencies)**

Apart from the necessary dependency in `requirements.txt`, some environments require extra dependencies and we list all of them here.

- `Python 3.7.5`
- `gym` https://github.com/openai/gym
- `gfootball` https://github.com/google-research/football
  
  (If want to use `football_5v5_malib`, put the `env/football_scenarios/malib_5_vs_5.py` file under folder like 
  `~/anaconda3/envs/env_name/lib/python3.x/site-packages/gfootball/scenarios`using environment `env_name` or
  `~/anaconda3/lib/python3.x/site-packages/gfootball/scenarios` using base environment.)
- `miniworld` https://github.com/maximecb/gym-miniworld#installation
- `minigrid` https://github.com/maximecb/gym-minigrid
- `Multi-Agent Particle Environment` https://www.pettingzoo.ml/mpe

  `pip install pettingzoo[mpe]==1.12.0`
  
  (Using `pip install 'pettingzoo[mpe]==1.12.0'` if you are using zsh.)
- `Overcooked-AI` https://github.com/HumanCompatibleAI/overcooked_ai
- `MAgent` https://www.pettingzoo.ml/magent
  
  (Using `pip install 'pettingzoo[magent]'` if you are using zsh; 
  Using render_from_log.py for MAgent local render)
- `SMARTS` https://gitee.com/mirrors_huawei-noah/SMARTS

  (Put repo `SMARTS` and `ai_lib` under the same folder.
  
  If not using smarts, comment out `from .smarts_jidi import *` in `env/__init__.py`.
  
  If want to use NGSIM scenario, download NGSIM scenario 
  here: https://www.dropbox.com/sh/fcky7jt49x6573z/AADUmqmIXhz_MfBcenid43hqa/ngsim?dl=0&subfolder_nav_tracking=1 
  and put the `ngsim` folder under `SMARTS/scenarios`.
  
  If not using smarts NGSIM, comment out `from .smarts_ngsim import *` in `env/__init__.py`.)
- `StartCraft II` https://github.com/deepmind/pysc2
- `olympics-running` https://github.com/jidiai/Competition_Olympics-Running

  (Notice: Put folder `olympics` and `jidi` under the same folder)
  
- `olympics-tablehockey olympics-football olympics-wrestling` https://github.com/jidiai/olympics_engine
  
  (Notice: Put repo `olympics_engine` and `jidi` under the same folder)
- `mujoco-py` https://github.com/openai/mujoco-py
- `Classic` https://www.pettingzoo.ml/classic 
  
  `pip install pettingzoo[classic]==1.12.0`
- `pip install rlcard==1.0.4`
  
  (Using `pip install 'pettingzoo[classic]==1.12.0'` if you are using zsh.)
- `gym-chinese-chess` https://github.com/bupticybee/gym_chinese_chess
- `Wilderness Scavenger` https://github.com/inspirai/wilderness-scavenger
- `REVIVE SDK` https://www.revive.cn/help/polixir-revive-sdk/text/introduction.html
- `FinRL` https://github.com/AI4Finance-Foundation/FinRL

  `pip install git+https://github.com/AI4Finance-Foundation/FinRL.git`
  
- `Torch 1.7.0` 可选
  - 支持提交Torch训练后的模型.pth附属文件

## **目录结构 (Code structure)**

```
|-- platform_lib
	|-- README.md
	|-- run_log.py		// 本地调试运行环境
	|-- examples	// 提交运行文件示例	需包含 my_controller 函数输出policy
	    |-- random.py  // 随机策略 需根据环境动作是否连续 调整 is_act_continuous 的值
	|-- replay		// render工具，用于非gym环境，打开replay.html上传run_log 存储的.json文件 
	|-- env		// 游戏环境 
	|	|-- simulators		// 模拟器
	|	|	|-- game.py
	|	|	|-- gridgame.py // 网格类模拟器接口
	|	|-- obs_interfaces		// observation 观测类接口
	|	|	|-- observation.py		// 目前支持Grid Vector
	|	|-- config.ini		// 相关配置文件
	|	|-- chooseenv.py 
	|	|-- snakes.py
	|	|-- gobang.py
	|	|-- reversi.py
	|	|-- sokoban.py
	|	|-- ccgame.py

```

## **平台提交说明**
1. 填写算法名称或描述，选择提交环境
2. 上传一个或多个文件。
- 其中必须包含一个运行文件，运行文件需包含`my_controller` 函数的一个`submission.py`文件。
- 附属文件支持`.pth` `.py`类型文件。大小不超过100M，个数不超过5个。 
