"""torch.profiler chrome trace of one forward for one length and mode."""
import argparse
import torch
from torch.profiler import ProfilerActivity, profile
import common

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, required=True)
p.add_argument("--mode", choices=["eager", "compile_stage"], required=True)
a = p.parse_args()

model = common.load_model()
if a.mode == "eager":
	common.ranges(model)
else:
	model = common.prepare(model, "compile_stage")
d = common.run_dir(a.length, a.mode)
path = d / "trace.json.gz"
x = common.load_input(a.length)
before = common.gpu_state()
status, error = "ok", ""
try:
	for _ in range(3):
		common.forward(model, x)
	torch.cuda.synchronize()
	with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
		common.forward(model, x)
		torch.cuda.synchronize()
	prof.export_chrome_trace(str(path))
except Exception as e:
	if not common.oom(e):
		raise
	status, error = "oom", str(e).splitlines()[0][:300]
after = common.gpu_state()
common.manifest(d, "trace", a.length, a.mode, x, before, after, status=status, error=error, trace=str(path) if status == "ok" else "")
print(f"trace {a.length} {a.mode} {common.prec}: {status}")
