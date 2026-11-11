"""pass 1: module call tree with in/out shapes for one forward pass, as a tab-indented txt (read back with common.read_tree)."""
import argparse
import torch
import common

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="cyp2b6_131k")
ap.add_argument("--fp32", action="store_true")
ap.add_argument("--check", action="store_true", help="parse the file back and compare counts")
ap.add_argument("-v", "--verbose", action="store_true")
a = ap.parse_args()


def desc(o):
	# compact, tab-free description of tensors inside nested args/outputs
	if torch.is_tensor(o):
		return f"{str(o.dtype)[6:]}[{','.join(map(str, o.shape))}]"
	if isinstance(o, dict):
		return "{" + ",".join(f"{k}:{desc(v)}" for k, v in o.items()) + "}"
	if isinstance(o, (list, tuple)):
		return "[" + ",".join(desc(v) for v in o) + "]"
	return type(o).__name__


model = common.load_model(a.fp32)
x, meta = common.load_input(a.name)
params = {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_modules()}
lines, stack, calls, hooks = [], [], {}, []


def pre(name):
	def hook(m, args, kwargs):
		calls[name] = calls.get(name, 0) + 1
		stack.append(len(lines))
		lines.append([len(stack) - 1, name or "root", type(m).__name__, calls[name], params[name], desc(args) + (" " + desc(kwargs) if kwargs else ""), ""])
	return hook


def post(m, args, out):
	lines[stack.pop()][6] = desc(out)


# step 1: pre/post hooks on every module
for n, m in model.named_modules():
	hooks.append(m.register_forward_pre_hook(pre(n), with_kwargs=True))
	hooks.append(m.register_forward_hook(post))

# step 2: one forward pass
common.run(model, x)
for h in hooks:
	h.remove()

# step 3: one line per module call, depth = leading tabs, fields tab-separated
common.results.mkdir(parents=True, exist_ok=True)
out = common.results / f"modules_{a.name.split('_')[-1]}.txt"
with open(out, "w") as f:
	f.write(f"# input {meta['chrom']}:{meta['start']}-{meta['end']} length {meta['length']} policy {model.dtype_policy} gpu {torch.cuda.get_device_name()} torch {torch.__version__}\n")
	f.write("# fields: name type call params in out\n")
	for l in lines:
		f.write("\t" * l[0] + "\t".join(map(str, l[1:])) + "\n")

if a.check:
	t = common.read_tree(out)
	n = len(lines)
	if len(t) != 1 or t[0]["name"] != "root":
		raise SystemExit("check failed: expected a single root")
	stack = list(t)
	while stack:
		node = stack.pop()
		n -= 1
		stack += node["children"]
	if n != 0:
		raise SystemExit("check failed: parsed node count differs")
if a.verbose:
	print(out, len(lines), "module calls")
