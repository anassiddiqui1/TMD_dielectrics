#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --account=su007-ndmh-gpu
#SBATCH --job-name=reduced_all

module purge; module load GCC/11.3.0 CUDA/11.7.0 OpenMPI/4.1.4 PyTorch/2.0-Miniconda3-4.12.0 SciPy-bundle

[ ! -z "$SLURM_ARRAY_TASK_ID" ] && export LOGSUFFIX="_"$SLURM_ARRAY_TASK_ID
export OMP_NUM_THREADS=32

python ~/mace/scripts/run_train.py \
    --name="MoWSSe_bi_radius_8_reduced_all_1_swa_800" \
    --train_file="MoWSSe_bi_train.xyz" \
    --valid_file="MoWSSe_bi_vald.xyz" \
    --compute_stress=True \
    --forces_weight=10.0 \
    --energy_weight=1.0 \
    --test_file="MoWSSe_bi_test.xyz" \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=8.0 \
    --batch_size=10 \
    --valid_batch_size=10 \
    --seed=123 \
    --max_num_epochs=1500 \
    --swa \
    --start_swa=800 \
    --loss="stress" \
    --MLP_irreps='64x0e' \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \

