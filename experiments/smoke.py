"""task 1: port vs the official alphagenome api on 131 kb of hg38 around cyp2b6; pearson r per output and tissue."""
import os, sys
import numpy as np
import torch
from alphagenome_pytorch.genome import GenomeSequenceSource, Interval
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
from alphagenome_pytorch.utils.sequence import sequence_to_onehot
import common

iv = Interval("chr19", 40939304, 41070376)
outputs = ["RNA_SEQ", "DNASE", "ATAC", "CAGE"]
tissues = {"liver": "UBERON:0002107", "lung": "UBERON:0002048", "brain": "UBERON:0000955"}
ref = common.data / "reference" / "smoke_cyp2b6_131k.npz"
cut = 0.99

# step 1: sequence; stop on a short window or any N
seq = GenomeSequenceSource(common.fasta).fetch_sequence(iv)
if len(seq) != iv.end - iv.start:
	raise SystemExit(f"got {len(seq)} bp, wanted {iv.end - iv.start}")
if seq.count("N"):
	raise SystemExit(f"{seq.count('N')} N bases in {iv}")

# step 2: api reference, fetched once
if not ref.exists():
	from alphagenome.models.v1 import dna_client
	api = dna_client.create(os.environ["ALPHAGENOME_API_KEY"], model_version=dna_client.ModelVersion.ALL_FOLDS)
	res = api.predict_sequence(seq, organism=dna_client.Organism.HOMO_SAPIENS, requested_outputs=[dna_client.OutputType[k] for k in outputs], ontology_terms=list(tissues.values()))
	save = {}
	for k in outputs:
		td = getattr(res, k.lower())
		save[f"{k}.values"] = td.values.astype(np.float32)
		save[f"{k}.res"] = np.array(td.resolution)
		save[f"{k}.name"] = td.metadata["name"].to_numpy(str)
		save[f"{k}.strand"] = td.metadata["strand"].to_numpy(str)
		save[f"{k}.curie"] = td.metadata["ontology_curie"].to_numpy(str)
	ref.parent.mkdir(parents=True, exist_ok=True)
	np.savez(ref, **save)
r = np.load(ref)

# step 3: port forward on the same sequence
model = common.load_model()
x = torch.from_numpy(sequence_to_onehot(seq, dtype=np.float32))[None].cuda()
out = common.forward(model, x)
cat = TrackMetadataCatalog.load_builtin("human")
print("#", common.env())

# step 4: same tracks by (name, strand) at the api resolution; r and max rel error per tissue
fail = []
print(f"{'output':8s} {'tissue':6s} {'res':>4s} {'tracks':>6s} {'r':>8s} {'max rel':>8s}")
for k in outputs:
	n = k.lower()
	res = int(r[f"{k}.res"])
	t = out[n][res][0].float().cpu().numpy()
	keys = [(m.track_name, m.get("strand")) for m in cat.get_tracks(n, num_tracks=t.shape[-1], strict=True)]
	idx = []
	for i in zip(r[f"{k}.name"], r[f"{k}.strand"]):
		if keys.count(i) != 1:
			raise SystemExit(f"{k}: track {i} found {keys.count(i)} times in port metadata, want 1")
		idx.append(keys.index(i))
	p = t[:, idx]
	for tis, c in tissues.items():
		m = r[f"{k}.curie"] == c
		if not m.any():
			print(f"{k:8s} {tis:6s} {res:4d} {0:6d}  FAIL no api tracks")
			fail.append(f"{k} {tis}")
			continue
		a, b = p[:, m].ravel().astype(np.float64), r[f"{k}.values"][:, m].ravel().astype(np.float64)
		pr = np.corrcoef(a, b)[0, 1]
		rel = np.abs(a - b).max() / np.abs(b).max()
		ok = pr >= cut
		print(f"{k:8s} {tis:6s} {res:4d} {int(m.sum()):6d} {pr:8.4f} {rel:8.4f}  {'ok' if ok else 'FAIL'}")
		if not ok:
			fail.append(f"{k} {tis}")

print("all passed" if not fail else f"{len(fail)} failed: {', '.join(fail)}")
sys.exit(1 if fail else 0)
