#!/bin/bash
# task 2: gpu type and whether torch, nsys and ncu work; writes nexus.txt
# run on a gpu node, from experiments/: bash nexus.sh
out=nexus.txt
mm="import torch; a = torch.randn(1024, 1024, device='cuda'); b = a @ a; torch.cuda.synchronize()"

date > $out
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> $out
echo "nsys path: $(which nsys)" >> $out
echo "ncu path: $(which ncu)" >> $out

# step 1: torch on the gpu
if python -c "$mm" > /tmp/torch.log 2>&1; then echo "torch: yes" >> $out; else echo "torch: no" >> $out; tail -3 /tmp/torch.log >> $out; fi

# step 2: nsys timeline of the matmul
if nsys profile -o /tmp/nexus_nsys --force-overwrite true python -c "$mm" > /tmp/nsys.log 2>&1; then echo "nsys: yes" >> $out; else echo "nsys: no" >> $out; tail -3 /tmp/nsys.log >> $out; fi

# step 3: ncu hardware counters on one kernel; ERR_NVGPUCTRPERM means counters are blocked
ncu --launch-count 1 python -c "$mm" > /tmp/ncu.log 2>&1
if grep -q "Speed Of Light" /tmp/ncu.log; then echo "ncu: yes" >> $out; else echo "ncu: no" >> $out; grep -i "error\|not found" /tmp/ncu.log | head -3 >> $out; fi

cat $out
