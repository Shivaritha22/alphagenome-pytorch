#!/bin/bash
# runs each argument as one command on the gpu node, in order; a failed command does not stop the rest
eval "$(conda shell.bash hook)"
conda activate /fs/nexus-scratch/shivrita/envs/ag312
module load cuda/13.1.1
export AG_DATA=/fs/nexus-scratch/shivrita/alphagenome/ag_data
export HF_HOME=/fs/nexus-scratch/shivrita/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/fs/nexus-scratch/shivrita/alphagenome/inductor_cache
cd ~/alphagenome-pytorch/experiments
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.sm,temperature.gpu --format=csv
for c in "$@"; do
	echo "== $(date +%T) $c"
	eval "$c" || echo "== failed ($?): $c"
done
echo "== $(date +%T) done"
