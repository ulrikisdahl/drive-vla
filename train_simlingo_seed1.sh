#!/bin/bash
#SBATCH --job-name=simlingo_train
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --time=26:00:00
#SBATCH --gres=gpu:a100:8
#SBATCH --constraint="gpu80g"
#SBATCH --cpus-per-task=32
#SBATCH --output=/cluster/home/ulrikyi/simlingo/results/logs/slurm.out  # File to which STDOUT will be written
#SBATCH --error=/cluster/home/ulrikyi/simlingo/results/logs/slurm.err   # File to which STDERR will be written
#SBATCH --partition=GPUQ
#SBATCH --open-mode=truncate
#SBATCH --account=share-ie-idi

# =============================================================================
# CONFIGURATION: Set to "true" to prestage LMDB to local NVMe before training
# =============================================================================
PRESTAGE_LMDB=false
LMDB_SOURCE_PATH="/cluster/projects/vc/data/ad/open/write-folder/simlingo/lmdb_dataset"
LMDB_LOCAL_PATH="/tmp/simlingo_lmdb"

# print info about current job
scontrol show job $SLURM_JOB_ID

source ~/.bashrc
module purge
module load Anaconda3/2024.02-1
conda activate simlingo

pwd
export CARLA_ROOT=/cluster/projects/vc/data/ad/open/write-folder/carla_0.9.15
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export WORK_DIR=/cluster/home/ulrikyi/simlingo
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}

export MASTER_ADDR=localhost
export NCCL_DEBUG=INFO

#Authenticate with Hugging Face to access gated models (expects HF_TOKEN to be exported before sbatch)
if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
else
    echo "Warning: HF_TOKEN is not set, Hugging Face downloads may fail." >&2
fi

export OMP_NUM_THREADS=32 # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.

# =============================================================================
# LMDB Prestaging: Copy LMDB database to local NVMe for faster I/O
# =============================================================================
if [[ "${PRESTAGE_LMDB}" == "true" ]]; then
    echo "Prestaging LMDB database to local NVMe..."
    echo "Source: ${LMDB_SOURCE_PATH}"
    echo "Destination: ${LMDB_LOCAL_PATH}"

    # Create local directory
    mkdir -p "${LMDB_LOCAL_PATH}"

    # Copy LMDB files with progress (data.mdb and lock.mdb)
    echo "Starting copy at $(date)"
    rsync -ah --info=progress2 "${LMDB_SOURCE_PATH}/data.mdb" "${LMDB_LOCAL_PATH}/"
    rsync -ah --info=progress2 "${LMDB_SOURCE_PATH}/lock.mdb" "${LMDB_LOCAL_PATH}/"
    echo "Copy completed at $(date)"

    # Verify files exist
    ls -lh "${LMDB_LOCAL_PATH}/"

    # Use local LMDB path and disable disk cache
    LMDB_OVERRIDE="data_module.base_dataset.use_lmdb=True data_module.base_dataset.lmdb_path=${LMDB_LOCAL_PATH} data_module.base_dataset.use_disk_cache=False"
    echo "Using local LMDB at: ${LMDB_LOCAL_PATH}"
else
    # Use disk cache (original behavior)
    LMDB_OVERRIDE="data_module.base_dataset.use_disk_cache=True data_module.base_dataset.dataset_cache_name=simlingo_cache data_module.base_dataset.dataset_cache_size_gb=1600"
    echo "Using disk cache (LMDB prestaging disabled)"
fi

WANDB__SERVICE_WAIT=300 python simlingo_training/train.py experiment=simlingo_seed1 data_module.batch_size=12 gpus=8 name=simlingo_full_4xA100_buckets ${LMDB_OVERRIDE}

#WANDB_SERVICE_WAIT=300 python simlingo_training/train.py experiment=simlingo_seed1 data_module.batch_size=12 gpus=8 name=simlingo_full_8xA100 resume=True resume_path=/cluster/home/ulrikyi/simlingo/outputs/2025_12_03_14_45_14_simlingo_full_8xA100/checkpoints/last.ckpt
