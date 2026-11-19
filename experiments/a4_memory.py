"""stage that owns the tensors alive at the memory peak, per length and mode, from the cuda memory snapshots."""
import csv, json, linecache, os, pickle, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

modes = ["eager", "compile_e2e"]
owners = ["resident"] + common.stages
gb = 2**30
# call-site text in model.py -> stage; specific patterns before the generic "head("
calls = [
	("self.encoder(", "encoder"),
	("self.organism_embed(", "transformer"),
	("block['pair_update']", "pairwise"),
	("block['attn_bias']", "pairwise"),
	("block['mha']", "transformer"),
	("block['mlp']", "transformer"),
	("self.decoder(", "decoder"),
	("self.embedder_128bp(", "embedders"),
	("self.embedder_1bp(", "embedders"),
	("self.embedder_pair(", "pairwise"),
	("self.splice_sites_junction_head(", "splice_junction"),
	("self.contact_maps_head(", "heads"),
	("self.splice_sites_classification_head(", "heads"),
	("self.splice_sites_usage_head(", "heads"),
	("head(", "heads"),
]
src = re.compile(r'File "([^"]*)", line (\d+)')


def is_model(f):
	return f.replace(os.sep, "/").endswith("alphagenome_pytorch/model.py")


def call_stage(f, n):
	# the line itself, or the start of a multi-line call still open at line n
	for k in range(n, max(n - 5, 0), -1):
		t = linecache.getline(f, k)
		if k < n:
			body = "".join(linecache.getline(f, j) for j in range(k, n))
			if body.count("(") <= body.count(")"):
				continue
		for p, s in calls:
			if p in t:
				return s
	return None


def owner(e):
	for fr in e.get("frames", []):
		if is_model(fr["filename"]):
			s = call_stage(fr["filename"], fr["line"])
			if s:
				return s
	# compiled code: original source stacks (fx traces per frame, then compile_context), each outermost first
	texts = [fr.get("fx_original_trace") or "" for fr in e.get("frames", [])] + [str(e.get("compile_context", ""))]
	for t in texts:
		for f, n in reversed(src.findall(t)):
			if is_model(f):
				s = call_stage(f, int(n))
				if s:
					return s
	return "other"


def replay(ev, before, stop):
	# returns allocated bytes, resident bytes and alive allocs {addr: index} after event stop
	alloc, res, alive = before, before, {}
	for i, e in enumerate(ev[:stop + 1]):
		if e["action"] == "alloc":
			alive[e["addr"]] = i
			alloc += e["size"]
		elif e["action"] == "free_completed":
			alloc -= e["size"]
			if alive.pop(e["addr"], None) is None:
				res -= e["size"]
	return alloc, res, alive


def peak(ev, before):
	alloc, best, at = before, before, -1
	for i, e in enumerate(ev):
		if e["action"] == "alloc":
			alloc += e["size"]
			if alloc > best:
				best, at = alloc, i
		elif e["action"] == "free_completed":
			alloc -= e["size"]
	return at


def frames_text(e):
	fs = [f"{os.path.basename(f['filename'])}:{f['line']} {f['name']}" for f in e.get("frames", [])[:6]]
	return " < ".join(fs) or "(no frames)"


rows, runs = [], {}
for L in common.lengths:
	for m in modes:
		d = common.raw / str(L) / f"{m}_bf16"
		if not (d / "snapshot.pickle").exists() or not (d / "memory.json").exists():
			print(f"skip {d}: no snapshot.pickle or memory.json")
			continue
		mem = json.load(open(d / "memory.json"))
		ev = pickle.load(open(d / "snapshot.pickle", "rb"))["device_traces"][0]
		ooms = [i for i, e in enumerate(ev) if e["action"] == "oom"]
		if mem["status"] == "oom":
			stop = ooms[0] if ooms else len(ev) - 1
		else:
			stop = peak(ev, mem["allocated_before_bytes"])
		alloc, res, alive = replay(ev, mem["allocated_before_bytes"], stop)
		by = {o: 0 for o in owners}
		by["resident"] = res
		big = None
		for i in alive.values():
			e = ev[i]
			by[owner(e)] += e["size"]
			if big is None or e["size"] > ev[big]["size"]:
				big = i
		runs[(L, m)] = {"status": mem["status"], "by": by, "alloc": alloc, "mem_peak": mem["peak_allocated_bytes"], "stop": stop, "big": big, "ev": ev[big] if big is not None else None}
		for o in owners:
			rows.append({"length": L, "mode": m, "owner": o, "gb": round(by[o] / gb, 4), "share": round(by[o] / alloc, 4) if alloc else 0, "peak_gb": round(alloc / gb, 4), "status": mem["status"]})

common.results.mkdir(exist_ok=True)
with open(common.results / "memory_peak.csv", "w", newline="") as f:
	w = csv.DictWriter(f, fieldnames=["length", "mode", "owner", "gb", "share", "peak_gb", "status"])
	w.writeheader()
	w.writerows(rows)

md = ["# A4 memory at peak (bf16)", ""]
md += ["Method: replay of the allocator trace in snapshot.pickle (device_traces[0], one recorded forward after 3 warm-ups).",
	"Allocated starts at memory.json allocated_before_bytes (weights + input, owner \"resident\"); alloc adds its size,",
	"free_completed subtracts it; frees of blocks allocated before recording reduce \"resident\". The peak is the alloc event",
	"with the highest running total; for oom runs the replay stops at the first oom event (or the end of the trace).",
	"Each alloc still alive there is owned by the innermost alphagenome_pytorch/model.py frame whose call-site line",
	"(or the start of the multi-line call it continues) matches a stage call (self.encoder(, block['mha'], head(, ...);",
	"if no frame matches, File \"...model.py\", line N entries in the fx_original_trace of its frames (compiled code), then in compile_context, are tried; else \"other\".",
	"\"attributed\" = non-resident bytes at the peak owned by a stage other than \"other\".", ""]
for m in modes:
	ls = [L for L in common.lengths if (L, m) in runs]
	if not ls:
		continue
	md += [f"## {m}", "", "| owner | " + " | ".join(f"{L}" + (" (oom)" if runs[(L, m)]["status"] == "oom" else "") for L in ls) + " |", "|---|" + "---:|" * len(ls)]
	for o in owners:
		md.append(f"| {o} (GB) | " + " | ".join(f"{runs[(L, m)]['by'][o] / gb:.3f}" for L in ls) + " |")
	md.append("| peak replay (GB) | " + " | ".join(f"{runs[(L, m)]['alloc'] / gb:.3f}" for L in ls) + " |")
	md.append("| peak memory.json (GB) | " + " | ".join(f"{runs[(L, m)]['mem_peak'] / gb:.3f}" for L in ls) + " |")
	att = []
	for L in ls:
		r = runs[(L, m)]
		act = r["alloc"] - r["by"]["resident"]
		a = act - r["by"]["other"]
		att.append(f"{a / gb:.3f} / {r['by']['other'] / gb:.3f} ({100 * a / act:.1f}%)" if act else "0 / 0")
	md.append("| attributed / other (GB) | " + " | ".join(att) + " |")
	md += ["", "Largest alive tensor at the peak:", ""]
	for L in ls:
		r = runs[(L, m)]
		if r["ev"] is None:
			md.append(f"- {L}: none")
			continue
		md.append(f"- {L}: event {r['big']} of peak event {r['stop']}, {r['ev']['size'] / gb:.3f} GB, owner {owner(r['ev'])}: {frames_text(r['ev'])}")
	md.append("")

fig, axs = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.5), squeeze=False)
for ax, m in zip(axs[0], modes):
	ls = [L for L in common.lengths if (L, m) in runs]
	x = range(len(ls))
	base = [0.0] * len(ls)
	for c, o in enumerate(owners):
		v = [runs[(L, m)]["by"][o] / gb for L in ls]
		ax.bar(x, v, bottom=base, label=o, color=plt.cm.tab10(c))
		base = [b + y for b, y in zip(base, v)]
	for i, L in enumerate(ls):
		if runs[(L, m)]["status"] == "oom":
			ax.text(i, base[i], "oom", ha="center", va="bottom")
	ax.set_xticks(list(x), [str(L) for L in ls], rotation=30)
	ax.margins(y=0.08)
	ax.set_title(m)
	ax.set_xlabel("length (bp)")
	ax.set_ylabel("GB at peak")
axs[0][-1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(common.results / "a4_memory.png", dpi=150)
md += ["![](a4_memory.png)", ""]
open(common.results / "a4_memory.md", "w").write("\n".join(md))
