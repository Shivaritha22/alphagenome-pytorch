"""Fixed golden input: CYP2B6 window from GRCh38.p13, one-hot via DeepMind's encoder."""
import argparse, datetime, hashlib, json, os, pathlib
import numpy as np
from alphagenome.data import genome
from alphagenome.io import fasta
from alphagenome_research.model import one_hot_encoder

AG = os.environ.get("AG_DATA", "/content/ag_data")
ap = argparse.ArgumentParser()
ap.add_argument("--fasta", default=f"{AG}/ref/GRCh38.p13.genome.fa")
ap.add_argument("--length", type=int, default=131072)
ap.add_argument("--out", default="inputs")
a = ap.parse_args()

iv = genome.Interval(chromosome="chr19", start=40991281, end=41018398).resize(a.length)
seq = fasta.FastaExtractor(a.fasta).extract(iv)
assert len(seq) == a.length, len(seq)
x = np.asarray(one_hot_encoder.DNAOneHotEncoder().encode(seq), dtype=np.float32)
assert x.shape == (a.length, 4), x.shape

name = f"cyp2b6_{a.length // 1000}k"
out = pathlib.Path(a.out); out.mkdir(exist_ok=True)
(out / f"{name}.txt").write_text(seq)
np.save(out / f"{name}_onehot.npy", x)
meta = {
    "name": name, "assembly": "GRCh38.p13", "fasta": a.fasta,
    "chrom": iv.chromosome, "start": iv.start, "end": iv.end, "length": len(seq),
    "n_bases": seq.upper().count("N"),
    "sha256_seq": hashlib.sha256(seq.encode()).hexdigest(),
    "sha256_onehot": hashlib.sha256(x.tobytes()).hexdigest(),
    "created": datetime.date.today().isoformat(),
}
json.dump(meta, open(out / f"{name}.json", "w"), indent=2)
print(iv, "| N bases:", meta["n_bases"], "| ->", out / f"{name}_onehot.npy")