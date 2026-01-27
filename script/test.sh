
#!/usr/bin/env bash
###
 # @Author: 颜峰 && bphengyan@163.com
 # @Date: 2023-05-19 17:19:11
 # @LastEditors: 颜峰 && bphengyan@163.com
 # @LastEditTime: 2023-05-22 09:54:42
 # @FilePath: /CO-MOT/tools/train.sh
 # @Description: 
 # 
 # Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
### 
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 bash script/test.sh 1 /RoboTron_Mani/exps/uvformer_calvin/run4/checkpoint_gripper_Temporal_UVFormer_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_fc_768_final_weights.pth 8020

# 打印所有指令
set -x

GPUS_NUM=$1
CKPT=$2
PORT=$3
export NODE_NUM=1
export NODE_RANK=0
export MASTER_ADDR=localhost
# Generate random port in range 29500-30000 (PyTorch recommended range)
export MASTER_PORT=$((29500 + RANDOM % 500))

export PYTHONPATH=$PYTHONPATH:/RoboTron-Mani/open_flamingo
export PYTHONPATH=$PYTHONPATH:/RoboTron-Mani/third_party/calvin/calvin_models
export PYTHONPATH=$PYTHONPATH:/RoboTron-Mani/third_party/calvin/calvin_env
export PYTHONPATH=$PYTHONPATH:/RoboTron-Mani/third_party/calvin/calvin_env/tacto_env
export PYTHONPATH=$PYTHONPATH:/RoboTron-Mani/
PYTHON=/miniconda3/envs/RBMM214/bin/python



$PYTHON robouniview/eval/policy_server.py --evaluate_from_checkpoint ${CKPT} --port ${PORT}
