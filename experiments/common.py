"""shared paths, model loading, the fixed input, the forward pass, the stage map, run dirs and manifests."""
import hashlib, json, os, pathlib, subprocess
import numpy as np
import torch

data = pathlib.Path(os.environ.get("AG_DATA", "/content/ag_data"))
here = pathlib.Path(__file__).resolve().parent
weights = data / "weights" / "model_all_folds.safetensors"
fasta = data / "ref" / "GRCh38.p13.genome.fa"
inputs = here / "inputs"
results = here / "results"
raw = pathlib.Path(os.environ.get("AG_RAW", "/fs/nexus-scratch/shivrita/alphagenome/results/a6000"))
name = "cyp2b6_1m"
lengths = [2048, 16384, 131072, 262144, 524288, 1048576]
fp32 = os.environ.get("AG_FP32") == "1"
prec = "fp32" if fp32 else "bf16"
modes = ["eager", "compile_e2e", "compile_stage"]
stages = ["encoder", "transformer", "pairwise", "decoder", "embedders", "heads", "splice_junction", "other"]


def fetch_weights():
	from huggingface_hub import hf_hub_download
	# public repo, no token
	if not weights.exists():
		hf_hub_download("gtca/alphagenome_pytorch", weights.name, local_dir=weights.parent)
	return weights


def load_model():
	from alphagenome_pytorch import AlphaGenome
	from alphagenome_pytorch.config import DtypePolicy
	# params fp32, compute bf16; AG_FP32=1 for gpus without bf16
	policy = DtypePolicy.full_float32() if fp32 else DtypePolicy.mixed_precision()
	return AlphaGenome.from_pretrained(fetch_weights(), dtype_policy=policy, device="cuda").eval().requires_grad_(False)


def load_input(length):
	meta = json.load(open(inputs / f"{name}.json"))
	x = np.load(inputs / f"{name}_onehot.npy")
	if hashlib.sha256(x.tobytes()).hexdigest() != meta["sha256_onehot"]:
		raise SystemExit(f"{name}: one-hot hash mismatch, rerun input.py")
	lo = (len(x) - length) // 2
	return torch.from_numpy(x[lo:lo + length])[None].cuda()


def forward(model, x):
	# predict() without the fp32 upcast of outputs
	with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32):
		return model(x, torch.zeros(1, dtype=torch.long, device=x.device))


def flatten(out, prefix=""):
	flat = {}
	for k, v in out.items():
		if isinstance(v, dict):
			flat.update(flatten(v, f"{prefix}{k}."))
		else:
			flat[f"{prefix}{k}"] = v
	return flat


def stage(module):
	if module.startswith("encoder"):
		return "encoder"
	if module.startswith("tower"):
		return "pairwise" if ".pair_update" in module or ".attn_bias" in module else "transformer"
	if module == "organism_embed":
		return "transformer"
	if module.startswith("embedder_pair"):
		return "pairwise"
	if module.startswith("decoder"):
		return "decoder"
	if module.startswith("embedder"):
		return "embedders"
	if module.startswith("splice_sites_junction"):
		return "splice_junction"
	if module.startswith(("heads", "contact_maps_head", "splice_")):
		return "heads"
	return "other"


def regions(model):
	# stage-level submodules; compile_stage compiles each one on its own
	names = ["encoder", "organism_embed"]
	for i, b in enumerate(model.tower.blocks):
		names += [f"tower.blocks.{i}.{k}" for k in b.keys() if b[k] is not None]
	names += ["decoder", "embedder_128bp", "embedder_1bp", "embedder_pair"]
	names += [f"heads.{k}" for k in model.heads.keys()]
	names += ["contact_maps_head", "splice_sites_classification_head", "splice_sites_usage_head", "splice_sites_junction_head"]
	return names


class Region(torch.nn.Module):
	# a compiled stage-level submodule; its named range sits outside the compiled code
	def __init__(self, name, m):
		super().__init__()
		self.name = name
		self.m = torch.compile(m, dynamic=False)

	def forward(self, *a, **k):
		with torch.profiler.record_function(self.name), torch.cuda.nvtx.range(self.name):
			return self.m(*a, **k)


def prepare(model, mode):
	# eager: the model; compile_e2e: one compiled graph; compile_stage: each region compiled, wrapped in a named range
	torch._dynamo.config.recompile_limit = 64
	if mode == "compile_e2e":
		return torch.compile(model, dynamic=False)
	if mode == "compile_stage":
		for n in regions(model):
			p, _, k = n.rpartition(".")
			parent = model.get_submodule(p)
			setattr(parent, k, Region(n, getattr(parent, k)))
	return model


def ranges(model):
	# eager: a profiler and nvtx range named after the module around every module call
	stack = []

	def pre(n):
		def f(m, a):
			r = torch.profiler.record_function(n)
			r.__enter__()
			stack.append(r)
			torch.cuda.nvtx.range_push(n)
		return f

	def post(m, a, o):
		torch.cuda.nvtx.range_pop()
		stack.pop().__exit__(None, None, None)

	for n, m in model.named_modules():
		m.register_forward_pre_hook(pre(n or "root"))
		m.register_forward_hook(post)


def oom(e):
	return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def run_dir(length, mode):
	d = raw / str(length) / f"{mode}_{prec}"
	d.mkdir(parents=True, exist_ok=True)
	return d


def smi(q):
	# one nvidia-smi query for the gpu this process uses
	i = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
	return subprocess.check_output(["nvidia-smi", "-i", i, f"--query-gpu={q}", "--format=csv,noheader,nounits"], text=True).strip()


def gpu_state():
	f = smi("clocks.sm,clocks.mem,temperature.gpu,power.draw").split(", ")
	return {"sm_mhz": f[0], "mem_mhz": f[1], "temp_c": f[2], "power_w": f[3]}


def gpu():
	return torch.cuda.get_device_name().split()[-1].lower()


def env():
	try:
		commit = subprocess.check_output(["git", "-C", str(here), "rev-parse", "--short", "HEAD"], text=True).strip()
	except Exception:
		commit = "unknown"
	return {
		"gpu": torch.cuda.get_device_name(),
		"gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
		"torch": torch.__version__,
		"cuda": torch.version.cuda,
		"driver": smi("driver_version"),
		"dtype": "fp32" if fp32 else "bf16 compute, fp32 params",
		"commit": commit,
	}


def manifest(d, item, length, mode, x, before, after, **kw):
	# one json per run: environment, clocks and temperature before and after, input hash
	meta = json.load(open(inputs / f"{name}.json"))
	m = {"item": item, "length": length, "mode": mode, "precision": prec, **env()}
	m["input_sha256"] = hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() if x is not None else ""
	m["input_seq_sha256"] = meta["sha256_crops"].get(str(length), "")
	m["slurm_job"] = os.environ.get("SLURM_JOB_ID", "")
	m["node"] = os.uname().nodename
	m["before"] = before
	m["after"] = after
	m.update(kw)
	json.dump(m, open(d / f"manifest_{item}.json", "w"), indent=1)
	return m
