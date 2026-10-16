"""shared paths, model loading, the fixed input and the tree reader for the port-side scripts."""
import hashlib, json, os, pathlib
import numpy as np
import torch

data = pathlib.Path(os.environ.get("AG_DATA", "/content/ag_data"))
here = pathlib.Path(__file__).resolve().parent.parent
weights = data / "weights" / "model_all_folds.safetensors"
fasta = data / "ref" / "GRCh38.p13.genome.fa"
results = here / "results"


def load_model(fp32=False):
	from huggingface_hub import hf_hub_download
	from alphagenome_pytorch import AlphaGenome
	from alphagenome_pytorch.config import DtypePolicy
	# public repo, no token; downloaded once per runtime
	if not weights.exists():
		hf_hub_download("gtca/alphagenome_pytorch", weights.name, local_dir=weights.parent)
	policy = DtypePolicy.full_float32() if fp32 else DtypePolicy.mixed_precision()
	return AlphaGenome.from_pretrained(weights, dtype_policy=policy, device="cuda").eval().requires_grad_(False)


def load_input(name):
	meta = json.load(open(here / "inputs" / f"{name}.json"))
	x = np.load(here / "inputs" / f"{name}_onehot.npy")
	if hashlib.sha256(x.tobytes()).hexdigest() != meta["sha256_onehot"]:
		raise SystemExit(f"{name}: one-hot hash mismatch, rerun port_input.py")
	return torch.from_numpy(x)[None].cuda(), meta


def run(model, x):
	# predict() without the final fp32 upcast; called as model(...) so root hooks fire
	with torch.no_grad(), torch.autocast("cuda", dtype=model.dtype_policy.compute_dtype, enabled=model.dtype_policy.compute_dtype != torch.float32):
		return model(x, torch.zeros(1, dtype=torch.long, device=x.device))


def read_tree(path):
	# tab depth -> nesting; fields: name type call params in out
	root, stack = [], []
	for line in open(path):
		if line.startswith("#"):
			continue
		d = len(line) - len(line.lstrip("\t"))
		f = line.rstrip("\r\n").lstrip("\t").split("\t")
		node = {"name": f[0], "type": f[1], "call": int(f[2]), "params": int(f[3]), "in": f[4], "out": f[5], "children": []}
		del stack[d:]
		(stack[-1]["children"] if stack else root).append(node)
		stack.append(node)
	return root
