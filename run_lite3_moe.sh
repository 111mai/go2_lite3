#!/usr/bin/env bash
set -euo pipefail

cd /mnt/A/hust_hbc/mai/lite3/go2_lite3

export CUDA_VISIBLE_DEVICES=1
export MPLCONFIGDIR=/tmp/lite3-mpl
export TORCH_EXTENSIONS_DIR=/tmp/lite3-torch-extensions
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/A/hust_hbc/mai/lite3/go2_lite3:/mnt/A/hust_hbc/mai/lite3/go2_lite3/rsl_rl:/mnt/A/hust_hbc/mai/isaacgym/python

/mnt/A/hust_hbc/mai/envs/lite3gym/bin/python -u \
legged_gym/scripts/train.py \
  --task=lite3_moe_cts \
  --num_envs=4096 \
  --max_iterations=150000 \
  --headless \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --run_name=main_moe_cts_4096_150k \
2>&1 | tee -a lite3_moe_cts_150k.log
