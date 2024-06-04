#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate foundationpose
which python
git checkout main
git pull
CUDA_VISIBLE_DEVICES=1 python run_pose_estimation_service.py