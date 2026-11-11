"""task 3: run the fixed input at every length; check shapes, values and centre agreement."""
import argparse, json, sys
import torch
import common

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 16384, 131072, 1048576])
ap.add_argument("-v", "--verbose", action="store_true")
a = ap.parse_args()


def rows(key, s):
	# expected sizes after the batch dim, from the output's resolution; None = not tied to length
	f = key.split(".")
	if f[0] == "contact_maps":
		return (s // 2048, s // 2048)
	if f[0] in ("splice_sites", "splice_site_usage"):
		return None if f[1] == "track_mask" else (s,)
	if f[0] == "splice_junctions":
		return None
	return (s // int(f[1]),)


def bounds(key):
	# value range each output must respect; None = unbounded (logits, contact maps)
	f = key.split(".")
	if f[-1] in ("1", "128") or key == "splice_junctions.pred_counts":
		return (0, float("inf"))
	if key in ("splice_sites.probs", "splice_site_usage.predictions"):
		return (0, 1)
	return None


def centre(key, t, s):
	# middle 2048 bp of a per-position output, kept on cpu for the cross-length check
	r = rows(key, s)
	if r is None or len(r) != 1:
		return None
	k = 2048 // (s // r[0])
	lo = (r[0] - k) // 2
	return t[0, lo:lo + k].float().cpu()


# step 1: model and the fixed input
model = common.load_model()
print("#", json.dumps(common.env(model)))
fail, sizes, mids = [], {}, {}

# step 2: one forward per length, every output checked on the gpu
for s in sorted(a.lengths):
	x = common.load_input(s)[0]
	out = common.flatten(common.forward(model, x))
	bad = []
	for k, t in out.items():
		r = rows(k, s)
		if r is not None and (t.shape[0] != 1 or tuple(t.shape[1:1 + len(r)]) != r):
			bad.append(f"{k} shape {list(t.shape)}, want rows {list(r)}")
		if t.is_floating_point():
			if not torch.isfinite(t).all():
				bad.append(f"{k} has {(~torch.isfinite(t)).sum().item()} nan/inf")
			b = bounds(k)
			if b is not None and (t.min().item() < b[0] or t.max().item() > b[1]):
				bad.append(f"{k} range [{t.min().item():.3g}, {t.max().item():.3g}], want {list(b)}")
			c = centre(k, t, s)
			if c is not None:
				mids.setdefault(k, {})[s] = c
		sizes.setdefault(k, {})[s] = list(t.shape)
		if a.verbose:
			print(f"  {k:40s} {str(t.dtype):16s} {list(t.shape)}")
	print(f"{s // 1024}kb: {len(out)} outputs, {'ok' if not bad else f'{len(bad)} failed'}")
	for i in bad:
		print("  FAIL", i)
	fail += bad
	del out, x
	torch.cuda.empty_cache()

# step 3: same outputs and same track counts at every length; fixed-size outputs identical
top = max(a.lengths)
for k, v in sizes.items():
	bad = None
	if len(v) != len(a.lengths):
		bad = f"{k} missing at some lengths"
	elif len({str(i[-1]) for i in v.values()}) != 1 or (rows(k, top) is None and len({str(i) for i in v.values()}) != 1):
		bad = f"{k} track count or shape changes with length: {v}"
	if bad:
		print("  FAIL", bad)
		fail.append(bad)

# step 4: centre 2048 bp agreement with the longest run (pearson r; reported, not a pass/fail)
if len(a.lengths) > 1:
	print("centre r vs", f"{top // 1024}kb:", "  ".join(f"{s // 1024}kb" for s in sorted(a.lengths) if s != top))
for k, v in (mids.items() if len(a.lengths) > 1 else []):
	r = []
	for s in sorted(v):
		if s != top:
			r.append(f"{torch.corrcoef(torch.stack([v[s].flatten(), v[top].flatten()]))[0, 1].item():.3f}")
	print(f"  {k:32s}", "  ".join(r))

print("all checks passed" if not fail else f"{len(fail)} checks failed")
sys.exit(1 if fail else 0)
