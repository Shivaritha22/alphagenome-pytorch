"""task 4: cut the fixed 1 mb hg38 window centred on cyp2b6, hash it, save the one-hot."""
import datetime, hashlib, json
import numpy as np
from alphagenome_pytorch.genome import GenomeSequenceSource, Interval
from alphagenome_pytorch.utils.sequence import sequence_to_onehot
import common

# cyp2b6 gene body, grch38, 0-based half-open
gene = ("chr19", 40991281, 41018398)
# sha256 of the 131 kb window used for the earlier jax run
old131k = "35f4645276292bbe5d3c0c0ddc06d157dc544f38b5c1c691c004ea57150fbec8"
size = 1048576
lengths = [2048, 16384, 131072, 524288, 1048576]

# step 1: 1 mb window centred on the gene (same rounding as deepmind's resize)
mid = (gene[1] + gene[2] + 1) // 2
iv = Interval(gene[0], mid - size // 2, mid + size // 2)

# step 2: fetch; refuse a short window or any unknown base
seq = GenomeSequenceSource(common.fasta).fetch_sequence(iv)
if len(seq) != size:
	raise SystemExit(f"got {len(seq)} bp, wanted {size}")
if seq.count("N"):
	raise SystemExit(f"{seq.count('N')} N bases in {iv}")

# step 3: one-hot and hashes, plus one hash per centred crop
x = sequence_to_onehot(seq, dtype=np.float32)
crops = {}
for t in lengths:
	lo = (size - t) // 2
	crops[t] = hashlib.sha256(seq[lo:lo + t].encode()).hexdigest()

# step 4: save; npy stays out of git, manifest goes in
common.inputs.mkdir(exist_ok=True)
np.save(common.inputs / f"{common.name}_onehot.npy", x)
meta = {
	"name": common.name,
	"assembly": "GRCh38.p13",
	"gene": "CYP2B6",
	"chrom": iv.chromosome,
	"start": iv.start,
	"end": iv.end,
	"length": size,
	"n_bases": 0,
	"sha256_seq": hashlib.sha256(seq.encode()).hexdigest(),
	"sha256_onehot": hashlib.sha256(x.tobytes()).hexdigest(),
	"sha256_crops": crops,
	"same_131k_as_jax_run": crops.get(131072) == old131k,
	"created": datetime.date.today().isoformat(),
}
json.dump(meta, open(common.inputs / f"{common.name}.json", "w"), indent=2)
print(iv, x.shape, "| N bases: 0 | 131k crop matches jax run:", meta["same_131k_as_jax_run"])
