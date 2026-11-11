"""shared paths, model loading, the fixed input and the forward pass for every task script."""
import hashlib, json, os, pathlib, subprocess
import numpy as np
import torch

data = pathlib.Path(os.environ.get("AG_DATA", "/content/ag_data"))
here = pathlib.Path(__file__).resolve().parent
weights = data / "weights" / "model_all_folds.safetensors"
fasta = data / "ref" / "GRCh38.p13.genome.fa"
inputs = here / "inputs"
results = here / "results"
name = "cyp2b6_1m"


def fetch_weights():
	from huggingface_hub import hf_hub_download
	# public repo, no token; once per runtime
	if not weights.exists():
		hf_hub_download("gtca/alphagenome_pytorch", weights.name, local_dir=weights.parent)
	return weights


def load_model():
	from alphagenome_pytorch import AlphaGenome
	from alphagenome_pytorch.config import DtypePolicy
	# params fp32, compute bf16, same as the public benchmark
	return AlphaGenome.from_pretrained(fetch_weights(), dtype_policy=DtypePolicy.mixed_precision(), device="cuda").eval().requires_grad_(False)


def load_input(length):
	meta = json.load(open(inputs / f"{name}.json"))
	x = np.load(inputs / f"{name}_onehot.npy")
	if hashlib.sha256(x.tobytes()).hexdigest() != meta["sha256_onehot"]:
		raise SystemExit(f"{name}: one-hot hash mismatch, rerun input.py")
	lo = (len(x) - length) // 2
	return torch.from_numpy(x[lo:lo + length])[None].cuda(), meta


def forward(model, x, **kw):
	# predict() minus its final fp32 upcast; called as model(...) so hooks fire
	with torch.no_grad(), torch.autocast("cuda", dtype=model.dtype_policy.compute_dtype):
		return model(x, torch.zeros(1, dtype=torch.long, device=x.device), **kw)


def flatten(out, prefix=""):
	# {"atac": {1: t, 128: t}} -> {"atac.1": t, "atac.128": t}
	flat = {}
	for k, v in out.items():
		if isinstance(v, dict):
			flat.update(flatten(v, f"{prefix}{k}."))
		else:
			flat[f"{prefix}{k}"] = v
	return flat


def env(model):
	try:
		commit = subprocess.check_output(["git", "-C", str(here), "rev-parse", "--short", "HEAD"], text=True).strip()
	except Exception:
		commit = "unknown"
	return {
		"gpu": torch.cuda.get_device_name(),
		"torch": torch.__version__,
		"cuda": torch.version.cuda,
		"dtype": str(model.dtype_policy),
		"commit": commit,
	}
