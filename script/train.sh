
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

# 打印所有指令
set -x

GPUS_NUM=$1
CONFIG_FILE=$2
NUM_SEQUENCES=120  # 3*8
WINDOW_SIZE=12


SAVE_DIR=/results/RoboTron_Mani


# eval "$('/mnt/data/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# conda activate /mnt/data/miniconda3/envs/RBMM214/


export PYTHONPATH=$PYTHONPATH:/RoboTron-Mani/open_flamingo
export PYTHONPATH=$PYTHONPATH:/RoboTron-Mani/third_party/calvin/calvin_models
export PYTHONPATH=$PYTHONPATH:/RoboTron-Mani/third_party/calvin/calvin_env
export PYTHONPATH=$PYTHONPATH:/RoboTron-Mani/third_party/calvin/calvin_env/tacto_env
export TORCH_HOME=/modelzoo/
export TORCHRUN=/miniconda3/envs/RBMM214/bin/torchrun
export SWANLAB_API_KEY=RLWjZrjWc6SV1vtz9loJs
# apt-get install openssh-server -y 

# apt --fix-broken install -y

# apt-get -y install libegl1-mesa libegl1
# apt-get -y install libgl1

# apt-get update -y 
# apt-get install -y  libegl1-mesa libegl1-mesa-dev

# apt install -y mesa-utils libosmesa6-dev llvm
# apt-get -y install meson
# apt-get -y build-dep mesa

# apt-get -y install freeglut3
# apt-get -y install freeglut3-dev

# apt-get install -y libgl1-mesa-dri

# apt update -y 
# apt install -y xvfb
# apt-get install patchelf -y
# apt-get install libc6-dev -y
# apt-get install -y libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev libqt5svg5-dev
# # echo "keyboard-configuration  keyboard-configuration/layout select English (US)" | sudo debconf-set-selections
# DEBIAN_FRONTEND=noninteractive apt-get install -y xorg

# apt-get install git -y 

# apt install -y  mesa-utils libosmesa6-dev llvm xorg libegl1-mesa libegl1-mesa-dev libegl1 libgles1 libgles2 libosmesa6 llvm-runtime  llvm-14 llvm-14-runtime mesa-utils-bin xserver-xorg libglu1-mesa xfonts-base x11-apps x11-session-utils x11-xkb-utils x11-xserver-utils xinit xfonts-utils xkb-data xorg-docs-core xinput xfonts-scalable  # libegl1 libgl1 libgl1-mesa-dri meson mesa
  # libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev libqt5svg5-dev

# apt install -y xvfb
# Xvfb :99 -screen 0 1024x768x16 &
# export DISPLAY=:99
# export MUJOCO_GL=osmesa
# export PYOPENGL_PLATFORM=osmesa

# export COPPELIASIM_ROOT=/project/robotic/RLBench/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
# export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# # export QT_DEBUG_PLUGINS=1

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/dri
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/anaconda3/envs/RBMM214/lib
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/dri/swrast_dri.so
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.8
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libxkbcommon-x11.so.0

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/robotic/Metaworld/mujoco210/bin

# export VK_ICD_FILENAMES=/.vulkan/icd.d/nvidia_icd.json
# export TERMINFO=/lib/terminfo
# export MS2_ASSET_DIR=/project/robotic/RoboUniview/third_party/ManiSkill/data

# export PYTHONPATH=$PYTHONPATH:/project/robotic/calvin/calvin_models
# export PYTHONPATH=$PYTHONPATH:/project/robotic/calvin/calvin_env
# export PYTHONPATH=$PYTHONPATH:/project/robotic/calvin/calvin_env/tacto_env




PY_ARGS=${@:2} # 第2个输入参数后边的值

# 脚本运行失败，报错
set -o pipefail
#sed -e  ：直接在指令列模式上進行 sed 的動作編輯；
OUTPUT_BASE=$(echo $2 | sed -e "s/config/exps/g" | sed -e "s/.yaml$//g")
OUTPUT_BASE=$SAVE_DIR/$OUTPUT_BASE
mkdir -p $OUTPUT_BASE

# cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
# echo "cluster spec is $cluster_spec"
# worker_list_command="import tools.json_parser as json_parser;print(json_parser.parse(\"$cluster_spec\", \"worker\"))"
# echo "worker list command is $worker_list_command"
# eval worker_list=`python -c "$worker_list_command"`
# echo "worker list is $worker_list"
# worker_strs=(${worker_list//,/ })
# master=${worker_strs[0]}
# echo "master is $master"
# master_strs=(${master//:/ })
# master_addr=${master_strs[0]}
# master_port=${master_strs[1]}
# echo "master address is $master_addr"
# echo "master port is $master_port"
# index_command="import tools.json_parser as json_parser;print(json_parser.parse(\"$cluster_spec\", \"index\"))"
# eval node_rank=`python -c "$index_command"`
# echo "node rank is $node_rank"
# dist_url="tcp://$master_addr:$master_port"
# echo "dist url is $dist_url"
# PYTHONPATH=$PYTHONPATH:../ \
# python tools/run_net.py \
#    --num_shards 8 \
#    --shard_id $node_rank \
#    --dist_url $dist_url \
#    --cfg configs/verb/MVIT_B_32x2_CONV.yaml

# MASTER_ADDR=${MASTER_ADDR:-$master_addr}
# MASTER_PORT=${MASTER_PORT:-$master_port}
# NODE_RANK=${NODE_RANK:-$node_rank}
# let "NNODES=GPUS/GPUS_PER_NODE"

# NODE_NUM=${#worker_strs[@]}  
# echo "node num is $NODE_NUM"

NODE_RANK=0
MASTER_ADDR=localhost
NODE_NUM=1

if ((NODE_RANK == 0)); then
  for RUN in $(seq 100); do
    ls $OUTPUT_BASE | grep run$RUN && continue
    OUTPUT_DIR=$OUTPUT_BASE/run$RUN
    mkdir $OUTPUT_DIR && break
  done

  # clean up *.pyc files
  rmpyc() {
    rm -rf $(find -name __pycache__)
    rm -rf $(find -name "*.pyc")
  }

  # run backup
  echo "Backing up to log dir: $OUTPUT_DIR"
  rmpyc && cp -r robouniview $OUTPUT_DIR
  cp $CONFIG_FILE $OUTPUT_DIR/$(basename $CONFIG_FILE)
  CONFIG_FILE=$OUTPUT_DIR/$(basename $CONFIG_FILE)

  echo " ...Done"

  # tar src to avoid future editing
  cleanup() {
    echo "Packing source code"
    rmpyc
    # tar -zcf models datasets util main.py engine.py eval.py submit.py --remove-files
    echo " ...Done"
  }

  # pushd $OUTPUT_DIR
  # log git status
  # echo "Logging git status"
  # git status > $OUTPUT_DIR/git_status
  # git rev-parse HEAD > $OUTPUT_DIR/git_tag
  # git diff > $OUTPUT_DIR/git_diff

else
  # 3 minutes
  sleep 180
  for RUN in $(seq 100); do
    ls $OUTPUT_BASE | grep run$RUN && continue
    let "ITERRUN=$RUN-1"
    OUTPUT_DIR=$OUTPUT_BASE/run$ITERRUN
    break
  done
fi


if true; then

  apt update -y 
  apt install -y xvfb
  apt install -y mesa-utils libosmesa6-dev llvm

  export PYTHON=/miniconda3/envs/calvin-yanfeng/bin/python
  export PYBULLET_EGL=0
  export PYBULLET_USE_TINY_RENDERER=1
  export PYOPENGL_PLATFORM=osmesa
  export MUJOCO_GL=osmesa
  export DISPLAY=:99
  export MUJOCO_GL=gl

  cp -r examples $OUTPUT_DIR

  # Start Calvin evaluator in background and register cleanup on script exit
  # Redirect output to a separate log file to avoid mixing with training logs
  $PYTHON examples/calvin/main.py --args.servers 0.0.0.0:8020,0.0.0.0:8021,0.0.0.0:8022,0.0.0.0:8023,0.0.0.0:8024,0.0.0.0:8025,0.0.0.0:8026,0.0.0.0:8027,0.0.0.0:8020,0.0.0.0:8021,0.0.0.0:8022,0.0.0.0:8023,0.0.0.0:8024,0.0.0.0:8025,0.0.0.0:8026,0.0.0.0:8027  --args.num_sequences=$NUM_SEQUENCES --args.window_size=$WINDOW_SIZE --args.no-only-once > "$OUTPUT_DIR/calvin_eval.log" 2>&1 &
  # $PYTHON examples/calvin/main.py --args.servers 0.0.0.0:8020,0.0.0.0:8020  --args.num_sequences=$NUM_SEQUENCES --args.window_size=$WINDOW_SIZE --args.no-only-once > "$OUTPUT_DIR/calvin_eval.log" 2>&1 &
  CALVIN_EVAL_PID=$!
  echo "Started Calvin evaluator in background (PID: $CALVIN_EVAL_PID, logs: $OUTPUT_DIR/calvin_eval.log)"
  
  # Ensure background process is killed when script exits
  trap "echo 'Cleaning up Calvin evaluator (PID: $CALVIN_EVAL_PID)'; kill $CALVIN_EVAL_PID 2>/dev/null" EXIT
fi



cd "$OUTPUT_DIR" || exit

PYTHONPATH=$PYTHONPATH:$OUTPUT_DIR


SWANLAB_API_KEY=${SWANLAB_API_KEY} $TORCHRUN --nproc_per_node=${GPUS_NUM} --nnodes ${NODE_NUM} --node_rank ${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port 29502 robouniview/train/train.py --config ${CONFIG_FILE} --save_dir $OUTPUT_DIR --report_to_wandb True |& tee -a $OUTPUT_DIR/output.log

# torchrun --nnodes=${NODE_NUM} --nproc_per_node=${GPUS_NUM} --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" --rdzv_backend=c10d  robouniview/train/train.py --config ${args} --save_dir $OUTPUT_DIR  |& tee -a $OUTPUT_DIR/output.log

# torchrun --nproc_per_node=${GPUS_NUM} --nnodes ${NODE_NUM} --node_rank ${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}  robouniview/train/train.py --config ${args} --save_dir $OUTPUT_DIR  |& tee -a $OUTPUT_DIR/output.log
