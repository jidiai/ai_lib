# JIdiAlgo V1.0 Framework

Now JidiAlgo V1.0 includes 12 single RL algorithms. 

We have tackled different envs by these algos and saved parameters in config files.

Come and play these games together~



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


---
## Start to play around^^

More information and results under each algorithm's folder.

>python main.py --scenario classic_CartPole-v0 --algo dqn --reload_config 

>python main.py --scenario classic_CartPole-v0 --algo ddqn --reload_config 

>python main.py --scenario classic_CartPole-v0 --algo duelingq --reload_config 

>python main.py --scenario classic_CartPole-v0 --algo pg --reload_config 

>python main.py --scenario classic_CartPole-v0 --algo ac --reload_config 

>python main.py --scenario classic_CartPole-v0 --algo ppo --reload_config 

>python main.py --scenario classic_CartPole-v0 --algo sac --reload_config 

>python main.py --scenario classic_Pendulum-v0 --algo td3 --reload_config 

>python main.py --scenario classic_CartPole-v0 --algo ddpg --reload_config 

>python main.py --scenario classic_MountainCar-v0 --algo dqn --reload_config 

>python main.py --scenario classic_MountainCar-v0 --algo ddqn --reload_config 

>python main.py --scenario classic_MountainCar-v0 --algo duelingq --reload_config 

>python main.py --scenario classic_Pendulum-v0 --algo sac --reload_config 

>python main.py --scenario classic_MountainCar-v0 --algo sac --reload_config 

>python main.py --scenario gridworld --algo sarsa --reload_config 

>python main.py --scenario gridworld --algo tabularq --reload_config
