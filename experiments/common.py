"""shared paths, model loading, the fixed input, the forward pass and the stage map."""
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
lengths = [2048, 16384, 131072, 524288, 1048576]
fp32 = os.environ.get("AG_FP32") == "1"


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
	if module == "embedder_pair":
		return "pairwise"
	if module.startswith("decoder"):
		return "decoder"
	if module.startswith("embedder"):
		return "embedders"
	if module.startswith(("heads", "contact_maps_head", "splice_")):
		return "heads"
	return "other"


def gpu():
	return torch.cuda.get_device_name().split()[-1].lower()


def env():
	try:
		commit = subprocess.check_output(["git", "-C", str(here), "rev-parse", "--short", "HEAD"], text=True).strip()
	except Exception:
		commit = "unknown"
	return {
		"gpu": torch.cuda.get_device_name(),
		"torch": torch.__version__,
		"cuda": torch.version.cuda,
		"dtype": "fp32" if fp32 else "bf16 compute, fp32 params",
		"commit": commit,
	}
