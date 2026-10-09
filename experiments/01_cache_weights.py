"""Download the weights once into DATA_ROOT/weights (see common.py). Later runs load from there.

The HF repo is public, so no token is needed. Re-run each Colab session unless AG_DATA
points at persistent storage.
"""
from huggingface_hub import hf_hub_download

import common


def main():
    if common.WEIGHTS_PATH.exists():
        print(f"already cached: {common.WEIGHTS_PATH} ({common.WEIGHTS_PATH.stat().st_size / 1e6:.0f} MB)")
        return
    common.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(common.WEIGHTS_REPO, common.WEIGHTS_FILE, local_dir=common.WEIGHTS_PATH.parent)
    print(f"saved: {common.WEIGHTS_PATH} ({common.WEIGHTS_PATH.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
