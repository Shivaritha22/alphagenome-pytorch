"""rebuild the fixed golden input with the port's own tools and refuse it unless both hashes match."""
import argparse, hashlib, json
import numpy as np
from alphagenome_pytorch.genome import GenomeSequenceSource, Interval
from alphagenome_pytorch.utils.sequence import sequence_to_onehot
import common

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="cyp2b6_131k")
ap.add_argument("-v", "--verbose", action="store_true")
a = ap.parse_args()

# step 1: coordinates and hashes from the committed manifest
meta = json.load(open(common.here / "inputs" / f"{a.name}.json"))
iv = Interval(meta["chrom"], meta["start"], meta["end"])

# step 2: fetch and encode with the port
seq = GenomeSequenceSource(common.fasta).fetch_sequence(iv)
x = sequence_to_onehot(seq, dtype=np.float32)

# step 3: byte-identical to the golden input or nothing
if hashlib.sha256(seq.encode()).hexdigest() != meta["sha256_seq"]:
	raise SystemExit("sequence hash mismatch")
if hashlib.sha256(x.tobytes()).hexdigest() != meta["sha256_onehot"]:
	raise SystemExit("one-hot hash mismatch")
np.save(common.here / "inputs" / f"{a.name}_onehot.npy", x)
if a.verbose:
	print(iv, x.shape, "hashes match")
