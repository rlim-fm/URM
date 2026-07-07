#!/bin/bash
#SBATCH --job-name=TARM-sudoku
#SBATCH --output=slurm/%j.log
#SBATCH --error=slurm/%j.err

# Usage: sbatch -p gpu --gres=gpu:pro6000:1 scripts/TARM_sudoku.sh

source .venv/bin/activate

run_name="TARM-sudoku"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

torchrun --nproc-per-node $SLURM_GPUS_ON_NODE pretrain.py \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    arch=tarm arch.loops=16 arch.H_cycles=2 arch.L_cycles=6 arch.num_layers=4 \
    epochs=50000 \
    eval_interval=2000 \
    lr=1e-4 puzzle_emb_lr=5e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=300 \
    +run_name=$run_name \
    +checkpoint_path=$checkpoint_path \
    +ema=True \
    +load_checkpoint=checkpoints/URM-sudoku/step_83325.pt \
    +load_filter_mismatched_shapes=True \
    +load_optimizer_state=False \
    +visualize=True \
