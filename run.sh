#!/bin/bash

# Get the directory of the run.sh script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (two levels up from the script directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"


python "$SCRIPT_DIR/train_simCLR_adaptive_update.py" \
    --experiment ASTrA_Training \
    --batch_size 512 \
    --interval_num 10 \
    --r1 0 \
    --sim_weight 0.5 \
    --ACL_DS \
    --gpu_ids "5" \
    --dataset "cifar10" \
    --policy_model_lr 0.1 \
    
    
