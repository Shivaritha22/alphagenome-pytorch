"""task 9: parallelism per stage from the module shapes of trace.py."""
import argparse, csv, math
import common

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=common.lengths)
a = ap.parse_args()

compute = ["Conv1d", "StandardizedConv1d", "MultiOrganismConv1d", "Linear", "MultiOrganismLinear"]
over = {
	"encoder": ["channels x positions, 1 bp down to 128 bp", "one output channel at one position, width-5 conv"],
	"transformer": ["heads x tokens (S/128)", "one query row over all tokens; one token through the mlp"],
	"pairwise": ["pair cells (S/2048)^2 x 128 channels", "one cell of the pair map"],
	"decoder": ["channels x positions, 128 bp up to 1 bp", "one output channel at one position, width-5 conv"],
	"embedders": ["channels x positions", "one position through a 1x1 projection"],
	"heads": ["tracks x positions", "one track at one position, dot product over the embedding"],
	"other": ["", ""],
}

for s in a.lengths:
	path = common.results / f"modules_{s}.tsv"
	if not path.exists():
		print(f"{s}: no modules tsv, skipped")
		continue
	# step 1: conv and linear calls per stage; largest output and its reduction
	calls = dict.fromkeys(over, 0)
	work = dict.fromkeys(over, 0)
	top = {k: ["", "", 0, ""] for k in over}
	for r in csv.DictReader(open(path), delimiter="\t"):
		if r["type"] not in compute:
			continue
		st = common.stage(r["name"])
		n = math.prod(int(i) for i in r["out"].split("x")) if r["out"] else 0
		calls[st] += 1
		work[st] += n * int(r["reduction"] or 0)
		if n > top[st][2]:
			top[st] = [r["name"], r["out"], n, r["reduction"]]

	# step 2: one row per stage
	with open(common.results / f"parallel_{s}.csv", "w", newline="") as fh:
		w = csv.writer(fh)
		w.writerow(["stage", "parallel over", "unit of work", "conv/linear calls", "largest op", "largest output shape", "output elements", "reduction per element", "total macs"])
		print(f"{s}:")
		for st in over:
			if calls[st] == 0:
				continue
			w.writerow([st, over[st][0], over[st][1], calls[st], top[st][0], top[st][1], top[st][2], top[st][3], work[st]])
			print(f"  {st:12s} calls {calls[st]:4d}  largest {top[st][1]:22s} {top[st][2]:>14,d} outputs x {top[st][3]:>5} in  macs {work[st]:.3e}")
