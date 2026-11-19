# experiments

Run on a Nexus GPU node, from `experiments/`, in this order:

```bash
conda activate /fs/nexus-scratch/shivrita/envs/alphagenome
pip install -e ~/alphagenome-pytorch pandas pyarrow huggingface_hub pyfaidx alphagenome
export AG_DATA=/fs/nexus-scratch/shivrita/alphagenome/ag_data
export HF_HOME=/fs/nexus-scratch/shivrita/hf_cache
export PIP_CACHE_DIR=/fs/nexus-scratch/shivrita/pip_cache
export ALPHAGENOME_API_KEY=...  # set in the shell, never in a file
wget -P $AG_DATA/ref https://storage.googleapis.com/alphagenome/reference/gencode/hg38/GRCh38.p13.genome.fa
wget -P $AG_DATA/ref https://storage.googleapis.com/alphagenome/reference/gencode/hg38/GRCh38.p13.genome.fa.fai
AG_FP32=0 python smoke.py  # bf16 vs api; first run saves $AG_DATA/reference/smoke_cyp2b6_131k.npz
AG_FP32=1 python smoke.py  # fp32 vs the same saved reference
python input.py            # 1 mb cyp2b6 window, one-hot and hashes
python lengths.py          # forward at every length, shape and value checks
```

## a6000 profiling (one forward, per stage, vs input length)

Collection on a GPU node, one (length, mode) per call, precision from `AG_FP32` (1 = fp32, else bf16); raw outputs go to
`$AG_RAW` (default `/fs/nexus-scratch/shivrita/alphagenome/results/a6000/<length>/<mode>_<precision>/`). `job.sh` runs a list of
these commands inside an sbatch job.

| script | item |
|---|---|
| c1_latency.py | latency (3 warm-up, 20 timed) and peak memory; eager, compile_e2e, compile_stage |
| c2_trace.py | torch.profiler trace with a named range per module (eager) or stage region (compile_stage) |
| c3_nsys.py | nsys timeline with nvtx ranges |
| c4_shapes.py | module shapes tsv |
| c5_flops.py | FlopCounterMode per region, aten op log |
| c6_memory.py | memory snapshot |
| c7_ceilings.py | measured matmul and copy ceilings (once) |

Analysis on CPU, in order: `a1_attribution.py`, then `a2_opmix.py a3_roofline.py a4_memory.py a5_parallel.py a6_compare.py`,
then `a7_scaling.py`, then `summary.py`. They write csv, md and png into `results/`.
