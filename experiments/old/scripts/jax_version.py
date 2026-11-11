"""Golden reference: raw JAX all_folds outputs (all heads, padding kept) for a fixed one-hot input."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse, datetime, json, pathlib, importlib.metadata as md
import numpy as np, jax, jax.numpy as jnp
from alphagenome_research.model import dna_model as rdm
from alphagenome_research.model.metadata import metadata as metadata_lib

AG = os.environ.get("AG_DATA", "/content/ag_data")
ap = argparse.ArgumentParser()
ap.add_argument("--input", default="inputs/cyp2b6_131k_onehot.npy")
ap.add_argument("--out", default=f"{AG}/golden/jax_allfolds_cyp2b6_131k.npz")
ap.add_argument("--manifest", default="golden/golden_manifest.json")
ap.add_argument("--save-embeddings", action="store_true")
a = ap.parse_args()

ORGS = [rdm.Organism.HOMO_SAPIENS, rdm.Organism.MUS_MUSCULUS]
# Empty settings = weights + track metadata only; no FASTA/GTF/calibration downloads.
model = rdm.create_from_huggingface(
    "all_folds", organism_settings={o: rdm.OrganismSettings() for o in ORGS})
_, apply_fn, *_ = rdm.create_model({o: metadata_lib.load(o) for o in ORGS})

x = jnp.asarray(np.load(a.input)[None])      # (1, L, 4)
org = jnp.zeros((1,), jnp.int32)             # 0 = human
preds = jax.device_get(jax.jit(apply_fn)(model._params, model._state, x, org))

def key_of(path):
    return ".".join(str(getattr(p, "key", getattr(p, "idx", getattr(p, "name", p)))) for p in path)

flat, rows = {}, []
for path, leaf in jax.tree_util.tree_flatten_with_path(preds)[0]:
    k = key_of(path)
    if k.startswith("embeddings") and not a.save_embeddings:
        continue
    v = np.asarray(leaf)
    if v.dtype.kind == "f" or "bfloat16" in str(v.dtype):
        v = v.astype(np.float32)
        rows.append((k, v.shape, bool(np.isnan(v).any()), float(v.min()), float(v.max())))
    else:
        rows.append((k, v.shape, False, None, None))
    flat[k] = v

pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(a.out, **flat)

try:
    commit = json.loads(md.distribution("alphagenome_research").read_text("direct_url.json"))["vcs_info"]["commit_id"]
except Exception:
    commit = "unknown"
inp = json.load(open(a.input.replace("_onehot.npy", ".json")))
pathlib.Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
json.dump({
    "weights": "google/alphagenome-all-folds", "framework": "jax", "jax": jax.__version__,
    "device": str(jax.devices()[0]), "alphagenome_research_commit": commit,
    "precision": "params fp32, compute bf16, output bf16 -> saved fp32",
    "organism_index": 0, "input": a.input, "input_sha256_onehot": inp["sha256_onehot"],
    "golden": a.out, "keys": {k: list(v.shape) for k, v in flat.items()},
    "created": datetime.datetime.now().isoformat(timespec="seconds"),
}, open(a.manifest, "w"), indent=2)

for k, s, nan, lo, hi in rows:
    print(f"{k:55s} {str(s):24s} nan={nan} min={lo} max={hi}")
print("saved ->", a.out)