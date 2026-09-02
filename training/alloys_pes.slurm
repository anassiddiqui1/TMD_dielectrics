#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --account=su007-ndmh-gpu
#SBATCH --job-name=mace_new

module purge; module load GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 RDKit/2020.03.3-Python-3.8.6 PyTorch/1.9.0 SciPy-bundle

[ ! -z "$SLURM_ARRAY_TASK_ID" ] && export LOGSUFFIX="_"$SLURM_ARRAY_TASK_ID
export OMP_NUM_THREADS=32

python ~/mace/scripts/run_train.py \
    --name="MoWSSe_MACE_new_version" \
    --train_file="MoWSSe_allxyT.xyz" \
    --valid_fraction=0.05 \
    --test_file="MoWSSe_allxyT_test.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --batch_size=10 \
    --valid_batch_size=10 \
    --seed=123 \
    --max_num_epochs=500 \
    --loss="stress" \
    --MLP_irreps='64x0e' \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
