

set -x

NUM_SEQUENCES=5
WINDOW_SIZE=12


apt update -y 
apt install -y xvfb
apt install -y mesa-utils libosmesa6-dev llvm

export PYTHON=/mnt/data/miniconda3/envs/calvin-yanfeng/bin/python
export PYBULLET_EGL=0
export PYBULLET_USE_TINY_RENDERER=1
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
export DISPLAY=:99
export MUJOCO_GL=gl

# $PYTHON examples/calvin/main.py --args.servers 0.0.0.0:8020,0.0.0.0:8021,0.0.0.0:8022,0.0.0.0:8023,0.0.0.0:8024,0.0.0.0:8025,0.0.0.0:8026,0.0.0.0:8027,0.0.0.0:8020,0.0.0.0:8021,0.0.0.0:8022,0.0.0.0:8023,0.0.0.0:8024,0.0.0.0:8025,0.0.0.0:8026,0.0.0.0:8027,0.0.0.0:8020,0.0.0.0:8021,0.0.0.0:8022,0.0.0.0:8023,0.0.0.0:8024,0.0.0.0:8025,0.0.0.0:8026,0.0.0.0:8027  --args.num_sequences=$NUM_SEQUENCES --args.window_size=$WINDOW_SIZE

$PYTHON examples/calvin/main.py --args.servers 0.0.0.0:8020,0.0.0.0:8020  --args.num_sequences=$NUM_SEQUENCES --args.window_size=$WINDOW_SIZE