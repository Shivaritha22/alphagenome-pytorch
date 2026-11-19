"""a2: operator mix per stage from the attributed gpu events of every traced run."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import common

cats = ["conv", "matmul", "attention", "norm", "elementwise", "cast/copy", "fused", "memcpy"]
color = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
precs = ["bf16", "fp32"]
conv_ops = {"convolution", "_convolution", "cudnn_convolution", "conv1d"}
mm_ops = {"mm", "addmm", "bmm", "matmul", "linear", "einsum", "baddbmm"}
score_ops = {"bmm", "baddbmm", "matmul", "einsum"}
attn_types = {"MHABlock", "RowAttentionBlock"}
copy_ops = {"copy_", "to", "_to_copy", "contiguous", "clone", "cat", "stack", "pad", "constant_pad_nd", "index", "index_select", "gather", "repeat_interleave", "flip"}
rules = [
	("memcpy", "cat is gpu_memcpy or gpu_memset"),
	("fused", "kernel name starts with `triton` (inductor-generated)"),
	("attention", "aten op `_softmax`/`softmax`/`scaled_dot_product*`, or kernel name has flash/fmha/efficient_attention/softmax, or aten op bmm/baddbmm/matmul/einsum whose innermost module is a `MHABlock` or `RowAttentionBlock` (the q.k and attn.v products; the q/k/v/out projections run in child `Linear` modules and stay matmul)"),
	("conv", "aten op convolution/_convolution/cudnn_convolution/conv1d, or kernel name has conv (not convert)/cudnn/fprop/dgrad/winograd/implicit_gemm"),
	("matmul", "aten op mm/addmm/bmm/matmul/linear/einsum/baddbmm, or kernel name has gemm/cutlass/nvjet/cublas/sm80_xmma/ampere_"),
	("norm", "module type contains `Norm` (`RMSBatchNorm`, `LayerNorm` incl. its fp32 casts and reductions), or aten op name has norm"),
	("cast/copy", "aten op " + "/".join(sorted(copy_ops)) + ", or kernel name has copy/cast/transpose"),
	("elementwise", "everything else (activations, adds, scaling, rope, reductions outside norm modules)"),
]


def classify(cat, kernel, op, typ):
	k = kernel.lower()
	o = op.removeprefix("aten::")
	if cat != "kernel":
		return "memcpy"
	if k.startswith("triton"):
		return "fused"
	if o in ("_softmax", "softmax") or "scaled_dot_product" in o or any(w in k for w in ("flash", "fmha", "efficient_attention", "softmax")):
		return "attention"
	if typ in attn_types and o in score_ops:
		return "attention"
	if o in conv_ops or any(w in k.replace("convert", "") for w in ("conv", "cudnn", "fprop", "dgrad", "winograd", "implicit_gemm")):
		return "conv"
	if o in mm_ops or any(w in k for w in ("gemm", "cutlass", "nvjet", "cublas", "sm80_xmma", "ampere_")):
		return "matmul"
	if "Norm" in typ or "norm" in o:
		return "norm"
	if o in copy_ops or any(w in k for w in ("copy", "cast", "transpose")):
		return "cast/copy"
	return "elementwise"


# module path -> module type; paths do not depend on length, so every shapes.tsv adds to one map
types = {}
for s in common.lengths:
	p = common.raw / str(s) / "eager_bf16" / "shapes.tsv"
	if p.exists():
		for r in csv.DictReader(open(p), delimiter="\t"):
			types[r["name"]] = r["type"]
if not types:
	print("no shapes.tsv: module types unknown, norm/attention rules by module type are off")

# step 1: ms per stage and category for every run
mix = {}
for s in common.lengths:
	for m in common.modes:
		for pr in precs:
			p = common.raw / str(s) / f"{m}_{pr}" / "kernels.csv.gz"
			if not p.exists():
				print(f"{s} {m}_{pr}: no kernels.csv.gz, skipped")
				continue
			df = pd.read_csv(p, dtype=str, keep_default_na=False)
			df["us"] = df["us"].astype(float)
			g = df.groupby(["stage", "module", "op", "cat", "kernel"], as_index=False)["us"].sum()
			g["category"] = [classify(c, k, o, types.get(mod, "")) for c, k, o, mod in zip(g["cat"], g["kernel"], g["op"], g["module"])]
			t = g.groupby(["stage", "category"])["us"].sum() / 1000
			mix[(s, m, pr)] = {st: {c: float(t.get((st, c), 0.0)) for c in cats} for st in common.stages}
if not mix:
	raise SystemExit("no kernels.csv.gz; run a1_attribution.py first")

# step 2: csv
common.results.mkdir(exist_ok=True)
with open(common.results / "op_mix.csv", "w", newline="") as f:
	w = csv.writer(f)
	w.writerow(["length", "mode", "precision", "stage", "category", "ms", "share_of_stage"])
	for (s, m, pr), d in mix.items():
		for st in common.stages:
			tot = sum(d[st].values())
			for c in cats:
				w.writerow([s, m, pr, st, c, round(d[st][c], 4), round(d[st][c] / tot, 4) if tot else 0])


# step 3: markdown
def table(d, share):
	head = "| stage | " + " | ".join(cats) + " | total ms |"
	out = [head, "|" + "---|" * (len(cats) + 2)]
	rows = [(st, d[st]) for st in common.stages] + [("all", {c: sum(d[st][c] for st in common.stages) for c in cats})]
	for st, v in rows:
		tot = sum(v.values())
		if tot == 0:
			continue
		cells = [f"{100 * v[c] / tot:.1f}" if share else f"{v[c]:.2f}" for c in cats]
		out.append(f"| {st} | " + " | ".join(cells) + f" | {tot:.2f} |")
	return out


md = ["# A2 operator mix per stage", ""]
md += ["Every GPU event of the profiled forward in `kernels.csv.gz` (from A1, already attributed to a stage and an innermost",
	"module path) is put in one category; ms per stage and category is the sum of event durations. Module types come from",
	"`shapes.tsv` (eager bf16) by module path. Rules, first match wins:", ""]
md += [f"{i + 1}. **{c}**: {r}" for i, (c, r) in enumerate(rules)]
md += ["", "Notes on this model:",
	"- attention is computed by hand (matmul/einsum + `nn.Softmax`, no sdpa), so the q.k and attn.v products are aten bmm",
	"  kernels inside `MHABlock`/`RowAttentionBlock`; the rope `einsum('bs,f->bsf')` in `MHABlock` also lands there (tiny).",
	"- in compile_stage the innermost module is the compiled region, so its categories are coarser: softmax, scaling,",
	"  norms and casts fuse into `triton` kernels (fused); only extern kernels (cuBLAS mm/bmm, cuDNN conv) keep their",
	"  category. Row attention bmm sits in the `pair_update` region (type `PairUpdateBlock`) and counts as matmul there.",
	"- conv includes cuDNN layout transforms launched under the convolution op.", ""]
md += ["![](a2_op_mix.png)", "", "## ms per stage and category", ""]
for (s, m, pr), d in mix.items():
	md += [f"### {s} {m} {pr}", ""] + table(d, False) + [""]
md += ["## share of stage time (%)", ""]
for (s, m, pr), d in mix.items():
	md += [f"### {s} {m} {pr}", ""] + table(d, True) + [""]
open(common.results / "a2_opmix.md", "w").write("\n".join(md))

# step 4: figure, rows = mode, columns = length
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
fm = [m for m in ("eager", "compile_stage") if any((s, m, "bf16") in mix for s in common.lengths)]
fl = [s for s in common.lengths if any((s, m, "bf16") in mix for m in fm)]
if fm:
	fig, axs = plt.subplots(len(fm), len(fl), figsize=(3.4 * len(fl) + 1.6, 3.2 * len(fm)), squeeze=False)
	for i, m in enumerate(fm):
		for j, s in enumerate(fl):
			ax = axs[i][j]
			ax.set_xlim(0, 100)
			ax.spines[["top", "right"]].set_visible(False)
			ax.set_title(f"{s} bp, {m}", color="#0b0b0b")
			if (s, m, "bf16") not in mix:
				ax.set_axis_off()
				continue
			d = mix[(s, m, "bf16")]
			tot = [sum(d[st].values()) for st in common.stages]
			base = [0.0] * len(common.stages)
			for c, col in zip(cats, color):
				v = [100 * d[st][c] / t if t else 0 for st, t in zip(common.stages, tot)]
				ax.barh(range(len(common.stages)), v, 0.7, left=base, color=col, edgecolor="#fcfcfb", linewidth=1)
				base = [a + b for a, b in zip(base, v)]
			ax.set_yticks(range(len(common.stages)), [f"{st} ({t:.1f} ms)" for st, t in zip(common.stages, tot)])
			ax.invert_yaxis()
			ax.set_xlabel("share of stage kernel time (%)")
	fig.legend([Patch(color=c) for c in color], cats, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 0.95))
	fig.tight_layout()
	fig.savefig(common.results / "a2_op_mix.png", dpi=150, bbox_inches="tight")
print(common.results / "op_mix.csv")
print(common.results / "a2_opmix.md")
