"""growth of per-stage time, flops and memory with input length, as log-log exponents."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import common

# from the architecture: per-bp stages linear, attention quadratic in L/128 tokens, pair grid (L/2048)^2
expected = {"encoder": "1", "transformer": "1-2", "pairwise": "2", "decoder": "1", "embedders": "1", "heads": "1", "splice_junction": "1", "other": "1"}
units = {"stage_ms": "ms", "stage_gflop": "GFLOP", "alive_gb": "GB", "latency_ms": "ms", "peak_allocated_gb": "GB"}
series = {}


def add(q, mode, p, s, length, v):
	series.setdefault((q, mode, p, s), {})[length] = v


f = common.results / "stage_times.csv"
if f.exists():
	for r in pd.read_csv(f).itertuples():
		add("stage_ms", r.mode, r.precision, r.stage, r.length, r.kernel_ms)
else:
	print(f"skip {f}")

for length in common.lengths:
	f = common.raw / str(length) / "eager_bf16" / "flops.tsv"
	if not f.exists():
		print(f"skip {f}")
		continue
	t = pd.read_csv(f, sep="\t")
	reg = t[t.module != "total"]
	tot = {s: 0.0 for s in common.stages}
	for r in reg.itertuples():
		tot[r.stage] += r.flops
	tot["other"] = float(t[t.module == "total"].flops.iloc[0]) - reg.flops.sum()
	for s in common.stages:
		add("stage_gflop", "eager", "bf16", s, length, tot[s] / 1e9)

f = common.results / "memory_peak.csv"
if f.exists():
	for r in pd.read_csv(f).itertuples():
		add("alive_gb", r.mode, "bf16", r.owner, r.length, r.gb)
else:
	print(f"skip {f}")

for length in common.lengths:
	for mode in common.modes:
		for p in ("bf16", "fp32"):
			f = common.raw / str(length) / f"{mode}_{p}" / "latency.json"
			if not f.exists():
				print(f"skip {f}")
				continue
			j = json.load(open(f))
			if j["status"] != "ok":
				print(f"skip {f}: {j['status']}")
				continue
			add("latency_ms", mode, p, "e2e", length, j["median_ms"])
			add("peak_allocated_gb", mode, p, "e2e", length, j["peak_allocated_gb"])


def points(v):
	# positive values in length order
	return [(length, v[length]) for length in common.lengths if length in v and v[length] > 0]


def fit(v):
	pts = [(length, y) for length, y in points(v) if length >= 131072]
	if len(pts) < 2:
		return np.nan
	x, y = np.log(np.array(pts)).T
	return np.polyfit(x, y, 1)[0]


rows = []
for (q, mode, p, s), v in series.items():
	local = {}
	pts = points(v)
	for (l0, y0), (l1, y1) in zip(pts, pts[1:]):
		local[l1] = np.log(y1 / y0) / np.log(l1 / l0)
	for length in common.lengths:
		if length in v:
			rows.append([q, mode, p, s, length, v[length], local.get(length, np.nan), expected.get(s, "")])
out = pd.DataFrame(rows, columns=["quantity", "mode", "precision", "stage", "length", "value", "local_exponent", "expected"])
out.to_csv(common.results / "scaling.csv", index=False)


def num(x):
	return "" if pd.isna(x) else f"{x:.3g}"


md = ["# A7 scaling with input length", "",
	"Each quantity y is measured at every input length L. The growth exponent is the slope of log(y) on log(L): "
	"1 means linear, 2 quadratic. scaling.csv holds the local exponent between each pair of consecutive lengths "
	"with data (on the row of the longer one). The tables give a least-squares fit over the lengths >= 131072 "
	"with data (needs >= 2 points; zero or missing values are left out).", "",
	"Expected from the architecture: encoder, decoder, embedders, heads and the rest work per base or per 128 bp bin, so 1. "
	"The transformer has attention quadratic in the L/128 tokens plus linear projections and MLPs, so between 1 and 2, nearing 2 "
	"as L grows. The pairwise stages work on (L/2048)^2 cells, so 2.", "",
	"Sources: stage_ms = kernel time per stage (stage_times.csv, A1); stage_gflop = FLOPs per stage (flops.tsv, eager bf16); "
	"alive_gb = GB alive at the peak per owner (memory_peak.csv, A4, bf16); latency_ms and peak_allocated_gb = end to end (latency.json).", "",
	"![](a7_scaling.png)", ""]
for (q, mode, p), g in out.groupby(["quantity", "mode", "precision"], sort=False):
	ls = sorted(g.length.unique())
	md += [f"## {q} ({units[q]}), {mode} {p}", "",
		"| stage | " + " | ".join(f"{length} bp" for length in ls) + " | fitted exponent | expected |",
		"|---|" + "---|" * (len(ls) + 2)]
	for s, h in g.groupby("stage", sort=False):
		v = dict(zip(h.length, h.value))
		md.append(f"| {s} | " + " | ".join(num(v.get(length, np.nan)) for length in ls) + f" | {num(fit(v))} | {expected.get(s, '')} |")
	md.append("")
(common.results / "a7_scaling.md").write_text("\n".join(md))

panels = [("stage_ms", "eager", "bf16", "stage time, eager bf16 (ms)"),
	("stage_ms", "compile_stage", "bf16", "stage time, compile_stage bf16 (ms)"),
	("stage_gflop", "eager", "bf16", "stage FLOPs (GFLOP)"),
	("alive_gb", "eager", "bf16", "GB alive at peak, eager"),
	("alive_gb", "compile_e2e", "bf16", "GB alive at peak, compile_e2e")]
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
for ax, (q, mode, p, title) in zip(axs.flat, panels):
	for (q2, m2, p2, s), v in series.items():
		pts = points(v)
		if (q2, m2, p2) == (q, mode, p) and pts:
			x, y = zip(*pts)
			ax.plot(x, y, marker="o", label=s)
	ax.set_xscale("log", base=2)
	ax.set_yscale("log")
	ax.set_xlabel("input length (bp)")
	ax.set_title(title)
	if ax.has_data():
		ax.legend(fontsize=7)
axs.flat[-1].axis("off")
fig.tight_layout()
fig.savefig(common.results / "a7_scaling.png", dpi=150)
print(f"wrote {common.results / 'scaling.csv'}, a7_scaling.md, a7_scaling.png")
