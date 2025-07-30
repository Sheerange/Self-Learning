#!/bin/sh
#SBATCH -J train
#SBATCH -o output1/log.%j.out
#SBATCH -e output1/log.%j.err
#SBATCH --partition=gpu-a100-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --gpus=1
python DRC_TCO.py --batch_size 64 --nw 20 --tco 0 --threshold_acc 100 --epochs 30 --dataset 'CUB311' --confidence 0.99 --lr 5e-3 --weight_decay 1e-4 --lambda_aug 0.3 --rewind_threshold 5 --epoch_start 4