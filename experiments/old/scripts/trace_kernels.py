"""pass 2: perfetto trace of one forward pass, every module wrapped in a named range so its kernels nest under it."""
import argparse
import torch
from torch.profiler import ProfilerActivity, profile, record_function
import common

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="cyp2b6_131k")
ap.add_argument("--fp32", action="store_true")
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("-v", "--verbose", action="store_true")
a = ap.parse_args()

model = common.load_model(a.fp32)
x, meta = common.load_input(a.name)
stack, hooks = [], []


def pre(name):
	def hook(m, args):
		stack.append(record_function(name or "root"))
		stack[-1].__enter__()
	return hook


def post(m, args, out):
	stack.pop().__exit__(None, None, None)


# step 1: a named range around every module call
for n, m in model.named_modules():
	hooks.append(m.register_forward_pre_hook(pre(n)))
	hooks.append(m.register_forward_hook(post))

# step 2: warm up outside the trace (cudnn autotune, allocator, lazy init)
for i in range(a.warmup):
	common.run(model, x)
torch.cuda.synchronize()

# step 3: one traced pass, synced so every kernel lands inside the trace
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
	common.run(model, x)
	torch.cuda.synchronize()
for h in hooks:
	h.remove()

common.results.mkdir(parents=True, exist_ok=True)
out = common.results / f"trace_{a.name.split('_')[-1]}.json"
prof.export_chrome_trace(str(out))
if a.verbose:
	print(out)
	print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
