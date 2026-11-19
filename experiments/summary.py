"""summary.md: the key tables from the a1-a7 csvs."""
import numpy as np
import pandas as pd
import common

r = common.results


def read(n):
	f = r / n
	if not f.exists():
		print(f"skip {f}")
		return pd.DataFrame()
	return pd.read_csv(f)


def table(df, index, cols, val, fmt):
	# rows index, columns cols, cells formatted, empty where missing
	if df.empty:
		return ["(no data)", ""]
	t = df.pivot_table(index=index, columns=cols, values=val, aggfunc="first")
	t = t.reindex(columns=[c for c in common.lengths if c in t.columns])
	# pipeline order for stage and owner rows
	order = ["resident"] + common.stages
	if set(t.index) <= set(order):
		t = t.reindex([i for i in order if i in t.index])
	out = [f"| {index} | " + " | ".join(str(c) for c in t.columns) + " |", "|---" * (len(t.columns) + 1) + "|"]
	for i, row in t.iterrows():
		out.append(f"| {i} | " + " | ".join("" if pd.isna(v) else fmt(v) for v in row) + " |")
	return out + [""]


def fit(g):
	# slope of log(value) on log(length) over lengths >= 131072
	g = g[(g.length >= 131072) & (g.value > 0)]
	if len(g) < 2:
		return np.nan
	return np.polyfit(np.log(g.length), np.log(g.value), 1)[0]


lat = read("latency.csv")
st = read("stage_times.csv")
rt = read("run_times.csv")
rf = read("roofline.csv")
mem = read("memory_peak.csv")
sc = read("scaling.csv")

md = ["# summary: one AlphaGenome forward on the RTX A6000", "", "Key findings in prose: [findings.md](findings.md).", "",
	"Details and method per step: [A1](a1_attribution.md), [A2](a2_opmix.md), [A3](a3_roofline.md), [A4](a4_memory.md), "
	"[A5](a5_parallel.md), [A6](a6_compare.md), [A7](a7_scaling.md). Status of the runs: [progress](progress.md).", ""]

md += ["## end-to-end latency, median ms (oom = out of memory)", ""]
if not lat.empty:
	lat["key"] = lat["mode"] + " " + lat.precision
	lat["cell"] = [s if s != "ok" else m for s, m in zip(lat.status, lat.median_ms)]
	md += table(lat, "key", "length", "cell", lambda v: v if isinstance(v, str) else f"{v:.1f}")
	md += ["## peak allocated GB", ""]
	md += table(lat[lat.status == "ok"], "key", "length", "peak_allocated_gb", lambda v: f"{v:.2f}")

for m in ["eager", "compile_stage"]:
	md += [f"## kernel time per stage, ms ({m} bf16)", ""]
	md += table(st[(st["mode"] == m) & (st.precision == "bf16")] if not st.empty else st, "stage", "length", "kernel_ms", lambda v: f"{v:.2f}")

md += ["## kernel sum vs gpu busy vs latency, ms (bf16)", ""]
if not rt.empty:
	b = rt[rt.precision == "bf16"]
	md += ["| length | mode | kernel | busy | latency | idle | unattributed |", "|---|---|---|---|---|---|---|"]
	for x in b.itertuples():
		md.append(f"| {x.length} | {x.mode} | {x.kernel_ms:.2f} | {x.busy_ms:.2f} | {x.latency_ms:.2f} | {x.idle_ms:.2f} | {x.unattributed_ms:.3f} |")
	md.append("")

md += ["## bound per stage, eager bf16 (A3; % of the relevant measured ceiling)", ""]
if not rf.empty:
	e = rf[(rf["mode"] == "eager") & (rf.precision == "bf16")].copy()
	e["cell"] = [f"{b} {p:.0f}%" for b, p in zip(e.bound, e.pct_peak)]
	md += table(e, "stage", "length", "cell", str)

md += ["## owner of the memory alive at the peak, GB (A4)", ""]
for m in ["eager", "compile_e2e"]:
	md += [f"### {m} bf16", ""]
	md += table(mem[mem["mode"] == m] if not mem.empty else mem, "owner", "length", "gb", lambda v: f"{v:.2f}")

md += ["## growth exponent with length (fit over L >= 131072, A7)", ""]
if not sc.empty:
	md += ["| stage | time eager bf16 | time compile_stage bf16 | FLOPs | memory at peak, eager | expected |", "|---|---|---|---|---|---|"]
	sel = [("stage_ms", "eager", "bf16"), ("stage_ms", "compile_stage", "bf16"), ("stage_gflop", "eager", "bf16"), ("alive_gb", "eager", "bf16")]
	for s in common.stages:
		v = []
		for q, m, p in sel:
			g = sc[(sc.quantity == q) & (sc["mode"] == m) & (sc.precision == p) & (sc.stage == s)]
			e = fit(g)
			v.append("" if np.isnan(e) else f"{e:.2f}")
		x = sc[sc.stage == s].expected.dropna()
		md.append(f"| {s} | " + " | ".join(v) + f" | {x.iloc[0] if len(x) else ''} |")
	md.append("")

open(r / "summary.md", "w").write("\n".join(md))
print(r / "summary.md")
