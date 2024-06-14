#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH --job-name=AlphaFlow

#SBATCH -p gpu
#SBATCH -A <INSERT ACCOUNT NAME HERE> 

#SBATCH --gres=gpu:v100:1 #CHANGE THIS IF ON ANOTHER CLUSTER WITHOUT v100s
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=3,4 # array job for multiple input csv files
#SBATCH --output=~/projects/sbatch/out/%x_%a.out

# OPTIONAL ARGS:
##SBATCH -C gpu32g
##SBATCH -d afterany:11707664

proj_dir="~/projects"
io_dir="${proj_dir}/data/davis/alphaflow_io/" #PATH TO INPUT_CSVs DIR

cd "${proj_dir}/alphaflow"
source ".venv-ds/bin/activate"
#                                   For LMA use only, defaults are:
export DEFAULT_LMA_Q_CHUNK_SIZE=512     # 1024
export DEFAULT_LMA_KV_CHUNK_SIZE=2048   # 4096

export CUDA_LAUNCH_BLOCKING=1

# Determine the number of GPUs available
world_size=$(nvidia-smi --list-gpus | wc -l)

# runs deepspeed if there are two or more GPUs requested
run_command() {
    if [[ $world_size -ge 2 ]]; then
        deepspeed --num_gpus $world_size predict_deepspeed.py --world_size $world_size $@
    else
        python predict.py $@
    fi
}

run_command --mode alphafold \
            --input_csv "${io_dir}/input_${SLURM_ARRAY_TASK_ID}.csv" \
            --msa_dir "${io_dir}/msa_dir" \
            --weights "weights/alphaflow_md_distilled_202402.pt" \
            --outpdb "${io_dir}/out_pdb_MD-distilled" \
            --samples 50 \
            --no_overwrite

