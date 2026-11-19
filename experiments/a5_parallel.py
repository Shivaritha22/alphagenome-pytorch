"""parallelism per stage: what each stage runs in parallel over, its conv/linear work and how its kernels fill the gpu."""
import csv, gzip, math
import common

sms = 84
# per stage: parallel over, unit of work, reduction per output; {L} = bp, {e} = L/64, {t} = L/128 tokens, {p} = L/2048 pair bins
text = {
	"encoder": [
		"positions x channels: convs at {L} positions (768 ch) down to {e} (1536 ch), max pool by 2 after each level, trunk {t} x 1536",
		"one output channel at one position of a conv (dna conv width 15, then width-5 standardized convs)",
		"4x15 = 60 (dna conv), then C_in x 5 = 3840 .. 7680",
	],
	"transformer": [
		"{t} tokens x 1536 ch; attention 8 query heads (1 shared k/v head) x {t} queries x {t} keys",
		"one token through a projection (q 1536->1024, k 1536->128, v 1536->192, out 1536->1536, mlp 1536->3072->1536); one query row of one head",
		"1536 or 3072 per projection; 128 per attention score, {t} keys per softmax and value sum",
	],
	"pairwise": [
		"pair map {p} x {p} cells x 128 ch ({t} tokens mean-pooled by 16); 5 pair updates, 9 attention biases",
		"one pair cell: 32-head q.k outer product, row attention over the {p} cells of its row, pair mlp 128->256->128; bias: one cell 128->8, repeated 16x16 to ({t})^2",
		"1536 (seq2pair q/k), 128 per head score, 32 (linear_pair), 128 per row-attention score, {p} per row softmax, 128/256 (pair mlp), 128 (bias)",
	],
	"decoder": [
		"positions x channels: {t} positions at 1536 ch, doubled 7 times (repeat by 2) to {L} bp at 768 ch",
		"one output channel at one position: width-5 conv_in and conv_out, 1x1 pointwise on the u-net skip",
		"C_in x 5 = 3840 .. 7680 (width 5), C_skip = 768 .. 1536 (1x1)",
	],
	"embedders": [
		"positions x channels: {t} x 3072 (128 bp), {L} x 1536 (1 bp, plus 3072->1536 skip at {t} repeated x128)",
		"one position through a 1x1 projection, then norm, organism add, gelu",
		"1536 (128 bp), 768 (1 bp), 3072 (skip)",
	],
	"heads": [
		"tracks x positions: 5 heads at {L} bp and {t} bins, chip heads at {t} bins only; contact maps ({p})^2 cells x 28; splice sites/usage {L} x 5/734",
		"one track at one position: organism-specific 1x1 dot product over the embedding",
		"1536 (1 bp), 3072 (128 bp), 128 (contact maps)",
	],
	"splice_junction": [
		"{L} positions x 768 hidden, then donor x acceptor pairs: 2 strands x 512 x 512 x 367 tissues",
		"one position through a 1x1 projection; one (donor, acceptor, tissue) count from rope-rotated gathered embeddings",
		"1536 (projection), 768 per donor/acceptor pair",
	],
	"other": [
		"glue in AlphaGenome.forward: transposes, organism add, top-k splice site selection over {L} x 4 probs",
		"elementwise or one sort/top-k segment",
		"none",
	],
}
cols = ["length", "stage", "calls", "largest_module", "largest_out", "largest_out_elements", "largest_reduction", "macs",
	"kernel_ms", "launches", "dominant_kernel", "dominant_ms", "dominant_share", "grid", "block", "blocks", "blocks_per_sm",
	"occupancy", "underfill_share", "parallel_over", "unit_of_work"]


def elems(s):
	# elements of the first tensor in "1x768x2048:bfloat16;..."
	if not s:
		return 0
	return math.prod(int(i) for i in s.split(";")[0].split(":")[0].split("x"))


def nblocks(g):
	# "128x1x1" -> 128, None if missing
	try:
		return math.prod(int(i) for i in g.split("x"))
	except ValueError:
		return None


def shapes(path):
	r = {s: {"calls": 0, "macs": 0, "top": None} for s in common.stages}
	for row in csv.DictReader(open(path), delimiter="\t"):
		if not row["reduction"]:
			continue
		d = r[row["stage"]]
		n = elems(row["out"])
		d["calls"] += 1
		d["macs"] += n * int(row["reduction"])
		if d["top"] is None or n > d["top"][2]:
			d["top"] = [row["name"], row["out"].split(";")[0], n, int(row["reduction"])]
	return r


def kernels(path):
	r = {s: {"us": 0.0, "n": 0, "by": {}, "known": 0.0, "under": 0.0} for s in common.stages}
	for row in csv.DictReader(gzip.open(path, "rt")):
		if row["cat"] != "kernel":
			continue
		d = r[row["stage"]]
		us = float(row["us"])
		d["us"] += us
		d["n"] += 1
		# time per kernel name, and per launch config within it
		k = d["by"].setdefault(row["kernel"], [0.0, {}])
		k[0] += us
		c = (row["grid"], row["block"], row["occupancy"])
		k[1][c] = k[1].get(c, 0.0) + us
		b = nblocks(row["grid"])
		if b is not None:
			d["known"] += us
			if b < sms:
				d["under"] += us
	return r


rows = []
for L in common.lengths:
	d = common.raw / str(L) / "eager_bf16"
	sp, kp = d / "shapes.tsv", d / "kernels.csv.gz"
	if not sp.exists() and not kp.exists():
		print(f"{L}: no shapes.tsv or kernels.csv.gz, skipped")
		continue
	if not sp.exists():
		print(f"{L}: no shapes.tsv, shape columns empty")
	if not kp.exists():
		print(f"{L}: no kernels.csv.gz, kernel columns empty")
	sh = shapes(sp) if sp.exists() else None
	kn = kernels(kp) if kp.exists() else None
	num = {"L": L, "e": L // 64, "t": L // 128, "p": L // 2048}
	for s in common.stages:
		r = dict.fromkeys(cols, "")
		r.update(length=L, stage=s, parallel_over=text[s][0].format(**num), unit_of_work=text[s][1].format(**num))
		if sh:
			a = sh[s]
			r.update(calls=a["calls"], macs=a["macs"])
			if a["top"]:
				r.update(largest_module=a["top"][0], largest_out=a["top"][1], largest_out_elements=a["top"][2], largest_reduction=a["top"][3])
		if kn and kn[s]["n"]:
			a = kn[s]
			name = max(a["by"], key=lambda k: a["by"][k][0])
			t, conf = a["by"][name]
			g, bl, occ = max(conf, key=conf.get)
			b = nblocks(g)
			r.update(kernel_ms=a["us"] / 1e3, launches=a["n"], dominant_kernel=name, dominant_ms=t / 1e3, dominant_share=t / a["us"],
				grid=g, block=bl, blocks="" if b is None else b, blocks_per_sm="" if b is None else b / sms, occupancy=occ,
				underfill_share=a["under"] / a["known"] if a["known"] else "")
		rows.append(r)

with open(common.results / "parallel.csv", "w", newline="") as f:
	w = csv.DictWriter(f, fieldnames=cols)
	w.writeheader()
	w.writerows(rows)


def fmt(v, spec):
	return "" if v == "" else format(v, spec)


def cell(v):
	return str(v).replace("|", "\\|")


sym = {"L": "L", "e": "L/64", "t": "L/128", "p": "L/2048"}
md = ["# A5 parallelism per stage", "", "## method", "",
	"- per length, eager bf16: R/<L>/eager_bf16/shapes.tsv (one row per module call) and kernels.csv.gz (one row per gpu event, from A1).",
	"- conv/linear calls = shapes rows with a nonempty reduction (Conv1d incl. StandardizedConv1d, Linear, MultiOrganismConv1d, MultiOrganismLinear).",
	"- largest op = the call with the most output elements (first output tensor); MACs = sum over calls of output elements x reduction.",
	"- MACs cover module-level convs/linears only; attention scores and value sums, the seq2pair outer product, row attention and the junction einsum are torch ops inside modules and are not counted here (see A3 for full flops).",
	"- kernel ms = sum of cat=kernel durations in the stage. dominant kernel = kernel name with the largest total time; its grid/block/occupancy are those of its longest-running launch config.",
	f"- blocks = grid x*y*z; blocks/SM = blocks / {sms} (A6000 SMs); occupancy = est. achieved occupancy % from the trace.",
	f"- underfill = time-weighted share of the stage's kernel time (launches with a known grid) spent in kernels with fewer than {sms} thread blocks, i.e. fewer blocks than SMs.",
	"", "## what each stage runs in parallel over", "",
	"| stage | parallel over | unit of work | reduction per output |", "|---|---|---|---|"]
for s in common.stages:
	md.append("| " + " | ".join([s] + [cell(i.format(**sym)) for i in text[s]]) + " |")
for L in common.lengths:
	rs = [r for r in rows if r["length"] == L]
	if not rs:
		continue
	md += ["", f"## L = {L}", "",
		"| stage | conv/linear calls | largest op | output shape | output elements | reduction | GMAC | kernel ms | dominant kernel | share | grid | block | blocks | blocks/SM | occupancy % | underfill |",
		"|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
	for r in rs:
		k = r["dominant_kernel"]
		md.append("| " + " | ".join(cell(i) for i in [r["stage"], r["calls"], r["largest_module"], r["largest_out"],
			fmt(r["largest_out_elements"], ","), r["largest_reduction"], fmt(r["macs"] and r["macs"] / 1e9, ".2f"),
			fmt(r["kernel_ms"], ".2f"), k[:60] + ("..." if len(k) > 60 else ""), fmt(r["dominant_share"], ".0%"),
			r["grid"], r["block"], fmt(r["blocks"], ","), fmt(r["blocks_per_sm"], ".2f"), r["occupancy"],
			fmt(r["underfill_share"], ".0%")]) + " |")
open(common.results / "a5_parallel.md", "w").write("\n".join(md) + "\n")
print(f"wrote {common.results / 'parallel.csv'} and {common.results / 'a5_parallel.md'}")
