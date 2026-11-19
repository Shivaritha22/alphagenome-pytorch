"""counting pass: flops per stage-level region with FlopCounterMode and a log of every aten op, eager."""
import argparse
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.flop_counter import FlopCounterMode
import common


def shapes(x):
	# broadcast (stride 0) dims written as 1, so a broadcast tensor counts its real elements
	return ";".join("x".join(str(1 if st == 0 else n) for n, st in zip(t.shape, t.stride())) + ":" + str(t.dtype).removeprefix("torch.") for t in tree_leaves(x) if isinstance(t, torch.Tensor))


class Log(TorchDispatchMode):
	# one row per aten op, attributed to the innermost running module
	def __init__(self, stack):
		super().__init__()
		self.stack = stack
		self.rows = []

	def __torch_dispatch__(self, func, types, args=(), kwargs=None):
		kwargs = kwargs or {}
		out = func(*args, **kwargs)
		m = self.stack[-1] if self.stack else "root"
		self.rows.append([m, common.stage(m), str(func), str(int(func.is_view or func == torch.ops.aten._unsafe_view.default)), shapes((args, kwargs)), shapes(out)])
		return out


def track(model, stack):
	# innermost module path on top of the stack
	def post(m, a, o):
		stack.pop()

	for n, m in model.named_modules():
		m.register_forward_pre_hook(lambda m, a, n=n or "root": stack.append(n))
		m.register_forward_hook(post)


def write(path, cols, rows):
	with open(path, "w") as f:
		f.write("\t".join(cols) + "\n")
		for r in rows:
			f.write("\t".join(map(str, r)) + "\n")


p = argparse.ArgumentParser()
p.add_argument("--length", type=int, required=True)
L = p.parse_args().length
d = common.run_dir(L, "eager")
model = common.load_model()
x = common.load_input(L)
stack = []
track(model, stack)
status, error, total = "ok", "", 0
before = common.gpu_state()
try:
	with FlopCounterMode(display=False) as fc, Log(stack) as log:
		common.forward(model, x)
	counts = fc.get_flop_counts()
	total = fc.get_total_flops()
	root = type(model).__name__
	rows = [[n, common.stage(n), sum(counts.get(f"{root}.{n}", {}).values())] for n in common.regions(model)]
	write(d / "flops.tsv", ["module", "stage", "flops"], rows + [["total", "", total]])
	write(d / "ops.tsv", ["module", "stage", "op", "view", "in", "out"], log.rows)
except Exception as e:
	if not common.oom(e):
		raise
	status, error = "oom", str(e).splitlines()[0][:300]
after = common.gpu_state()
common.manifest(d, "flops", L, "eager", x, before, after, status=status, error=error, total_flops=total)
