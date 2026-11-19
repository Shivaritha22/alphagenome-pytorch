"""module input/output shapes, one row per module call, from one eager forward."""
import argparse
import torch
from alphagenome_pytorch.heads import MultiOrganismConv1d, MultiOrganismLinear
import common

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, required=True)
a = p.parse_args()


def tensors(v):
	# every tensor in v as "1x768x2048:bfloat16", recursing into dicts/lists/tuples
	if torch.is_tensor(v):
		return ["x".join(str(i) for i in v.shape) + ":" + str(v.dtype).removeprefix("torch.")]
	if isinstance(v, dict):
		v = list(v.values())
	if isinstance(v, (list, tuple)):
		return [s for i in v for s in tensors(i)]
	return []


def reduction(m):
	# inputs summed per output element
	if isinstance(m, torch.nn.Conv1d):
		return m.in_channels * m.kernel_size[0] // m.groups
	if isinstance(m, MultiOrganismConv1d):
		return m.in_channels
	if isinstance(m, (torch.nn.Linear, MultiOrganismLinear)):
		return m.in_features
	return ""


rows = []
stack = []


def pre(n):
	# reserve the row at call time so rows are in call order
	def f(m, args, kwargs):
		r = [n, type(m).__name__, common.stage(n), int(not any(m.children())), ";".join(tensors([args, kwargs])), "", reduction(m), sum(q.numel() for q in m.parameters(recurse=False))]
		rows.append(r)
		stack.append(r)
	return f


def post(m, args, kwargs, out):
	stack.pop()[5] = ";".join(tensors(out))


model = common.load_model()
for n, m in model.named_modules():
	m.register_forward_pre_hook(pre(n or "root"), with_kwargs=True)
	m.register_forward_hook(post, with_kwargs=True)
d = common.run_dir(a.length, "eager")
x = common.load_input(a.length)
before = common.gpu_state()
status, error = "ok", ""
try:
	common.forward(model, x)
	torch.cuda.synchronize()
except Exception as e:
	if not common.oom(e):
		raise
	status, error = "oom", str(e).splitlines()[0][:300]
after = common.gpu_state()
if status == "ok":
	with open(d / "shapes.tsv", "w") as f:
		f.write("name\ttype\tstage\tleaf\tin\tout\treduction\tparams\n")
		for r in rows:
			f.write("\t".join(str(i) for i in r) + "\n")
common.manifest(d, "shapes", a.length, "eager", x, before, after, status=status, error=error, rows=len(rows) if status == "ok" else 0)
print(f"shapes {a.length} eager {common.prec}: {status}, {len(rows)} rows")
