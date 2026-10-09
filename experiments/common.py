"""Shared helpers for the profiling scripts: paths, model loading, input, env info.

Import-only (no CLI). Scripts live in experiments/, run as `python experiments/NN_*.py` from the repo root.
Paths: weights and outputs live on Drive; stage_map.json lives next to the scripts (committed in the repo).
"""
import json
import os
import platform
import subprocess
from pathlib import Path

import torch

_COLAB_DRIVE = Path("/content/drive/MyDrive/alphagenome")
DRIVE_ROOT = Path(os.environ.get("AG_DRIVE") or (_COLAB_DRIVE if _COLAB_DRIVE.parent.exists() else Path.home() / ".cache" / "alphagenome"))
WEIGHTS_REPO = "gtca/alphagenome_pytorch"
WEIGHTS_FILE = "model_all_folds.safetensors"
WEIGHTS_PATH = DRIVE_ROOT / "weights" / WEIGHTS_FILE
OUT_DIR = Path(os.environ.get("AG_OUT") or DRIVE_ROOT / "out")
SCRIPTS_DIR = Path(__file__).resolve().parent
STAGE_MAP = SCRIPTS_DIR / "stage_map.json"
UPSTREAM_BASE = "72268c0"  # upstream commit this fork was profiled from; model code is not modified


def tag(length: int) -> str:
    """16384 -> '16kb'. Goes in every output filename."""
    return f"{length // 1024}kb"


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device=None, dtype_policy=None):
    """Load the model from the Drive cache in eval mode with grads off.

    AG_RANDOM_INIT=1 builds an untrained model instead (tooling smoke tests only;
    the structure and shapes are identical, the numbers mean nothing).
    """
    from alphagenome_pytorch import AlphaGenome

    device = device or default_device()
    if os.environ.get("AG_RANDOM_INIT"):
        print("WARNING: AG_RANDOM_INIT set -> random weights, smoke test only")
        model = (AlphaGenome(dtype_policy=dtype_policy) if dtype_policy else AlphaGenome()).to(device)
    else:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"{WEIGHTS_PATH} not found; run 01_cache_weights.py first")
        model = AlphaGenome.from_pretrained(WEIGHTS_PATH, dtype_policy=dtype_policy, device=device)
    return model.eval().requires_grad_(False)


def make_input(length: int, device, seed: int = 0):
    """One-hot DNA (1, length, 4) float32, NLC. Validates the length first,
    because forward() does not and a bad length crashes mid-pass."""
    from alphagenome_pytorch.model import validate_sequence_length

    validate_sequence_length(length)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (1, length), generator=g)
    return torch.nn.functional.one_hot(idx, 4).float().to(device)


def run_forward(model, x):
    """The one call every script profiles: forward(), all heads, all resolutions, human."""
    organism = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    with torch.no_grad():
        return model(x, organism)


# ---- shape descriptions --------------------------------------------------

def _t(t) -> str:
    return f"{str(t.dtype).replace('torch.', '')}{list(t.shape)}"


def desc(o, limit=8):
    """Compact JSON-able description of tensors in an arbitrary structure."""
    if torch.is_tensor(o):
        return _t(o)
    if isinstance(o, dict):
        return {str(k): desc(v, limit) for k, v in list(o.items())[:limit]}
    if isinstance(o, (list, tuple)):
        items = [desc(v, limit) for v in o[:limit]]
        return [d for d in items if d is not None] or None
    return None


def iter_tensors(o, prefix=""):
    """Yield (dotted_key, tensor) for every tensor in a nested output."""
    if torch.is_tensor(o):
        yield prefix, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from iter_tensors(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from iter_tensors(v, f"{prefix}[{i}]")


# ---- environment info (goes into every manifest / meta) ------------------

def port_commit() -> str:
    try:
        import importlib.metadata as md

        direct = json.loads(md.distribution("alphagenome-pytorch").read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id")
        if commit:
            return commit
        url = direct.get("url", "")
        if url.startswith("file:///"):
            path = url[len("file:///"):] if os.name == "nt" else url[len("file://"):]
            return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "port_commit": port_commit(),
        "upstream_base": UPSTREAM_BASE,
        "weights": "random-init" if os.environ.get("AG_RANDOM_INIT") else WEIGHTS_FILE,
    }
    try:
        info["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        info["driver"] = None
    return info


def load_stage_map() -> dict:
    """stage -> [module prefixes]; keys starting with '_' are notes and are dropped."""
    if not STAGE_MAP.exists():
        return {}
    raw = json.loads(STAGE_MAP.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}
