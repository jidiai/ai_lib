#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export PYTHONPATH=$SCRIPT_DIR


#python main_marl.py --config expr/leduc_poker/expr_q_learning_marl.yaml
#python main_pbt.py --config expr/leduc_poker/expr_q_learning_psro.yaml
#python main_pbt.py --config expr/kuhn_poker/expr_q_learning_psro.yaml
#python main_marl.py --config expr/kuhn_poker/expr_q_learning_marl.yaml
python main_marl.py --config expr/cartpole/expr_dqn_marl.yaml


