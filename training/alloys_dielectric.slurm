#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --account=su007-ndmh-gpu
#SBATCH --job-name=ndata_all_raman

module purge; module load GCC/11.3.0 CUDA/11.7.0 OpenMPI/4.1.4 PyTorch/2.0-Miniconda3-4.12.0 SciPy-bundle

[ ! -z "$SLURM_ARRAY_TASK_ID" ] && export LOGSUFFIX="_"$SLURM_ARRAY_TASK_ID
export OMP_NUM_THREADS=32

python ~/mace/scripts/run_train.py \
    --name="MoWSSe_susram_ndata_all_weights_10_1_new" \
    --train_file="all.xyz" \
    --valid_fraction=0.05 \
    --test_file="all.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --model="AtomicDipolesMACE" \
    --polarizability_key="susceptibility_tensor" \
    --polarizability_deriv_key="raman_tensor" \
    --dipole_weight=0.0 \
    --dipole_deriv_weight=0.0 \
    --polarizability_weight=10.0 \
    --polarizability_deriv_weight=1.0 \
    --forces_key=None \
    --energy_key=None \
    --stress_key=None \
    --r_max=5.0 \
    --hidden_irreps='128x0e + 128x1o + 128x2e' \
    --batch_size=10 \
    --valid_batch_size=10 \
    --seed=123 \
    --max_num_epochs=500 \
    --loss="dipole_polarizability_deriv" \
    --compute_forces=False \
    --error_table='DipolePolarizabilityDerivRMSE' \
    --restart_latest \
    --device=cuda \
    --MLP_irreps='64x0e' \
