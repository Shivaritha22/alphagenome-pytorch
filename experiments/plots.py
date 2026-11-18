"""task 8: stage share vs length; operator mix per stage."""
import argparse, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=common.lengths)
a = ap.parse_args()

stages = ["encoder", "transformer", "pairwise", "decoder", "embedders", "heads", "other"]
ops = ["conv", "matmul", "attention", "norm", "elementwise", "cast/copy", "memcpy"]
color = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
label = {2048: "2 kb", 16384: "16 kb", 131072: "131 kb", 262144: "262 kb", 524288: "524 kb", 1048576: "1 Mb"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

# step 1: per-stage and per-stage-per-op microseconds for every traced length
done = []
share = {}
mix = {}
for s in a.lengths:
	if not (common.results / f"kernels_{s}.csv").exists():
		continue
	done.append(s)
	share[s] = dict.fromkeys(stages, 0.0)
	mix[s] = {st: dict.fromkeys(ops, 0.0) for st in stages}
	for r in csv.DictReader(open(common.results / f"kernels_{s}.csv")):
		share[s][r["stage"]] += float(r["us"])
		mix[s][r["stage"]][r["op"]] += float(r["us"])
if not done:
	raise SystemExit("no kernels_<length>.csv; run attribute.py first")

# step 2: stacked stage share vs length
fig, ax = plt.subplots(figsize=(7, 4))
x = [label.get(s, str(s)) for s in done]
base = [0.0] * len(done)
for st, c in zip(stages, color):
	v = [100 * share[s][st] / sum(share[s].values()) for s in done]
	ax.bar(x, v, 0.6, bottom=base, color=c, edgecolor="#fcfcfb", linewidth=1.5, label=st)
	base = [i + j for i, j in zip(base, v)]
ax.set_ylabel("share of gpu kernel time (%)")
ax.set_xlabel("input length")
ax.set_ylim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
fig.savefig(common.results / "stage_share.png", dpi=200)

# step 3: operator mix inside each stage, one panel per length
fig, axs = plt.subplots(1, len(done), figsize=(3.2 * len(done) + 1.5, 3.6), sharey=True, squeeze=False)
for ax, s in zip(axs[0], done):
	base = [0.0] * len(stages)
	for o, c in zip(ops, color):
		v = [100 * mix[s][st][o] / max(sum(mix[s][st].values()), 1e-9) for st in stages]
		ax.barh(stages, v, 0.6, left=base, color=c, edgecolor="#fcfcfb", linewidth=1.5, label=o)
		base = [i + j for i, j in zip(base, v)]
	ax.set_title(label.get(s, str(s)), color="#0b0b0b")
	ax.set_xlim(0, 100)
	ax.set_xlabel("share of stage time (%)")
	ax.spines[["top", "right"]].set_visible(False)
axs[0][0].invert_yaxis()
axs[0][-1].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
fig.savefig(common.results / "op_mix.png", dpi=200)
print(common.results / "stage_share.png")
print(common.results / "op_mix.png")
