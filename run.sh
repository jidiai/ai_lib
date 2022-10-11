#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export PYTHONPATH=$SCRIPT_DIR


python light_malib/main_marl.py --config light_malib/expr/leduc_poker/expr_q_learning_marl.yaml
#python light_malib/main_pbt.py --config light_malib/expr/leduc_poker/expr_q_learning_psro.yaml
#python light_malib/main_pbt.py --config light_malib/expr/kuhn_poker/expr_q_learning_psro.yaml


