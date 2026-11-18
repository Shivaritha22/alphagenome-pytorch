"""task 6: one profiled eager forward per length, a named range around every module call; module shapes per length."""
import argparse, json
import torch
from torch.profiler import ProfilerActivity, profile, record_function
import common

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=common.lengths)
ap.add_argument("--warmup", type=int, default=2)
a = ap.parse_args()

model = common.load_model()
print("#", json.dumps(common.env()))
stack = []
log = None


def dims(v):
	# shape of the first tensor in v
	if torch.is_tensor(v):
		return "x".join(str(i) for i in v.shape)
	if isinstance(v, dict):
		v = list(v.values())
	if isinstance(v, (list, tuple)):
		for i in v:
			d = dims(i)
			if d:
				return d
	return ""


def reduction(m):
	# inputs summed per output element
	if hasattr(m, "in_features"):
		return m.in_features
	if isinstance(m, torch.nn.Conv1d):
		return m.in_channels * m.kernel_size[0] // m.groups
	if hasattr(m, "in_channels"):
		return m.in_channels
	return ""


def pre(name):
	def hook(m, args):
		stack.append(record_function(name))
		stack[-1].__enter__()
	return hook


def post(name):
	def hook(m, args, out):
		stack.pop().__exit__(None, None, None)
		if log is not None:
			log.append([name, type(m).__name__, dims(args), dims(out), reduction(m)])
	return hook


# step 1: hooks on every module
for n, m in model.named_modules():
	m.register_forward_pre_hook(pre(n or "root"))
	m.register_forward_hook(post(n or "root"))

common.results.mkdir(exist_ok=True)
for s in a.lengths:
	try:
		x = common.load_input(s)
		# step 2: one pass logging module shapes, then warm-up
		log = []
		common.forward(model, x)
		rows = log
		log = None
		for i in range(a.warmup):
			common.forward(model, x)
		torch.cuda.synchronize()
		# step 3: one profiled pass
		with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
			common.forward(model, x)
			torch.cuda.synchronize()
	except torch.cuda.OutOfMemoryError:
		print(f"{s}: out of memory, skipped")
		stack.clear()
		log = None
		x = None
		torch.cuda.empty_cache()
		continue
	prof.export_chrome_trace(str(common.results / f"trace_{s}.json.gz"))
	with open(common.results / f"modules_{s}.tsv", "w") as fh:
		fh.write("name\ttype\tin\tout\treduction\n")
		for r in rows:
			fh.write("\t".join(str(i) for i in r) + "\n")
	print(f"{s}: trace_{s}.json.gz, modules_{s}.tsv ({len(rows)} module calls)")
	x = None
	prof = None
	torch.cuda.empty_cache()
