"""Download the weights once into the Drive cache. Later runs load from there only.

Colab: mount Drive first (from google.colab import drive; drive.mount('/content/drive')).
The HF repo is public, so no token is needed.
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
