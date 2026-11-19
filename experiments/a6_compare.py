"""per-stage and end-to-end speedups of bf16 over fp32 and of compile over eager."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import common

precs = ["bf16", "fp32"]
rows = []
for L in common.lengths:
	for m in common.modes:
		for p in precs:
			f = common.raw / str(L) / f"{m}_{p}" / "latency.json"
			if not f.exists():
				print(f"skip {f}")
				continue
			r = json.load(open(f))
			rows.append({"length": L, "mode": m, "precision": p, "status": r["status"], "median_ms": r["median_ms"],
				"q1_ms": r["q1_ms"], "q3_ms": r["q3_ms"], "peak_allocated_gb": r["peak_allocated_gb"], "peak_reserved_gb": r["peak_reserved_gb"]})
cols = ["length", "mode", "precision", "status", "median_ms", "q1_ms", "q3_ms", "peak_allocated_gb", "peak_reserved_gb"]
lat = pd.DataFrame(rows, columns=cols)
common.results.mkdir(exist_ok=True)
lat.to_csv(common.results / "latency.csv", index=False)

# ms per (length, stage, mode, precision); stage "total" = summed kernel time, "end_to_end" = latency median
ms = {}
f = common.results / "stage_times.csv"
if f.exists():
	st = pd.read_csv(f)
	for (L, m, p), g in st.groupby(["length", "mode", "precision"]):
		for s, k in zip(g["stage"], g["kernel_ms"]):
			ms[(L, s, m, p)] = k
		ms[(L, "total", m, p)] = g["kernel_ms"].sum()
else:
	print(f"skip {f}")
for r in rows:
	if r["status"] == "ok":
		ms[(r["length"], "end_to_end", r["mode"], r["precision"])] = r["median_ms"]

# (comparison, mode_or_precision, stages, a key, b key); speedup = b / a
stages = common.stages + ["total"]
pairs = []
for m in ["eager", "compile_stage"]:
	pairs.append(("bf16_vs_fp32", m, stages + ["end_to_end"], (m, "bf16"), (m, "fp32")))
pairs.append(("bf16_vs_fp32", "compile_e2e", ["end_to_end"], ("compile_e2e", "bf16"), ("compile_e2e", "fp32")))
for p in precs:
	pairs.append(("compile_vs_eager", p, stages + ["end_to_end"], ("compile_stage", p), ("eager", p)))
	pairs.append(("compile_e2e_vs_eager", p, ["end_to_end"], ("compile_e2e", p), ("eager", p)))
out = []
for c, mp, ss, ka, kb in pairs:
	for L in common.lengths:
		for s in ss:
			a, b = ms.get((L, s) + ka), ms.get((L, s) + kb)
			# zero kernel time (stage absent) has no speedup
			if a and b:
				out.append({"length": L, "stage": s, "comparison": c, "mode_or_precision": mp, "a_ms": a, "b_ms": b, "speedup": b / a})
cmp = pd.DataFrame(out, columns=["length", "stage", "comparison", "mode_or_precision", "a_ms", "b_ms", "speedup"])
cmp.to_csv(common.results / "compare.csv", index=False)


def table(df, index, rows):
	# pipe table, rows x lengths, empty cell where missing
	t = df.pivot(index=index, columns="length", values="speedup").reindex(index=rows, columns=common.lengths)
	lines = [f"| {index} | " + " | ".join(str(L) for L in common.lengths) + " |", "|---" * (len(common.lengths) + 1) + "|"]
	for i, r in t.iterrows():
		lines.append(f"| {i} | " + " | ".join("" if pd.isna(v) else f"{v:.2f}" for v in r) + " |")
	return lines


def cell(L, m, p):
	g = lat[(lat["length"] == L) & (lat["mode"] == m) & (lat["precision"] == p)]
	if g.empty:
		return ["", ""]
	r = g.iloc[0]
	if r["status"] == "oom":
		return ["oom", "oom"]
	return [f"{r['median_ms']:.1f} [{r['q1_ms']:.1f}-{r['q3_ms']:.1f}]", f"{r['peak_allocated_gb']:.2f}"]


md = ["# A6 comparisons: bf16 vs fp32, compile vs eager", "", "## method", "",
	"- latency: median [q1-q3] of 20 timed forwards after 3 warm-ups (latency.json); peak = torch.cuda.max_memory_allocated in GB (2**30).",
	"- per-stage time: summed GPU kernel time per stage from the profiled forward (A1 stage_times.csv); total = sum over stages.",
	"- bf16_vs_fp32: speedup = fp32 ms / bf16 ms, per mode (eager, compile_stage per stage; every mode end to end).",
	"- compile_vs_eager: speedup = eager ms / compile_stage ms, per precision; compile_e2e_vs_eager: eager / compile_e2e latency.",
	"- end_to_end rows use latency medians; speedup > 1 means the first-named variant is faster. missing or oom runs are empty cells.",
	"", "## latency", "",
	"| length | mode | bf16 median [q1-q3] (ms) | bf16 peak (GB) | fp32 median [q1-q3] (ms) | fp32 peak (GB) |", "|---|---|---|---|---|---|"]
for L in common.lengths:
	for m in common.modes:
		md.append(f"| {L} | {m} | " + " | ".join(cell(L, m, "bf16") + cell(L, m, "fp32")) + " |")
for c, mp, title in [("bf16_vs_fp32", "eager", "bf16 vs fp32, eager (fp32 / bf16)"),
	("bf16_vs_fp32", "compile_stage", "bf16 vs fp32, compile_stage (fp32 / bf16)"),
	("compile_vs_eager", "bf16", "compile_stage vs eager, bf16 (eager / compile_stage)"),
	("compile_vs_eager", "fp32", "compile_stage vs eager, fp32 (eager / compile_stage)")]:
	md += ["", f"## speedup: {title}", ""]
	md += table(cmp[(cmp["comparison"] == c) & (cmp["mode_or_precision"] == mp)], "stage", stages + ["end_to_end"])
e2e = cmp[cmp["stage"] == "end_to_end"].copy()
e2e["comparison"] = e2e["comparison"] + " " + e2e["mode_or_precision"]
names = [f"{c} {mp}" for c, mp, ss, ka, kb in pairs]
md += ["", "## speedup: end to end (latency medians)", ""] + table(e2e, "comparison", names)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for i, (c, sub) in enumerate([("bf16_vs_fp32", ["eager", "compile_stage"]), ("compile_vs_eager", precs)]):
	for j, mp in enumerate(sub):
		for k, s in enumerate(stages):
			g = cmp[(cmp["comparison"] == c) & (cmp["mode_or_precision"] == mp) & (cmp["stage"] == s)]
			if not g.empty:
				ax[i].plot(g["length"], g["speedup"], marker="o", ls=["-", "--"][j], color="k" if s == "total" else f"C{k}",
					label=f"{s} ({mp})")
	ax[i].axhline(1, color="grey", lw=0.8)
	ax[i].set_xscale("log", base=2)
	ax[i].set_yscale("log", base=2)
	ax[i].set_xlabel("sequence length (bp)")
	ax[i].set_ylabel("speedup (x)")
	ax[i].legend(fontsize=6, ncol=2)
ax[0].set_title("bf16 / fp32 per stage (fp32 ms / bf16 ms)")
ax[1].set_title("compile_stage / eager per stage (eager ms / compile ms)")
fig.tight_layout()
fig.savefig(common.results / "a6_compare.png", dpi=150)
md += ["", "![](a6_compare.png)", ""]
open(common.results / "a6_compare.md", "w").write("\n".join(md))
print(f"wrote {len(lat)} latency rows, {len(cmp)} compare rows")
