"""task 3: forward pass at every length; shapes, nan/inf, value ranges, centre agreement."""
import argparse, json, sys
import torch
import common

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 16384, 131072, 1048576])
ap.add_argument("-v", "--verbose", action="store_true")
a = ap.parse_args()


def rows(key, s):
	# s / resolution per output; none for fixed-size outputs
	f = key.split(".")
	if f[0] == "contact_maps":
		return [s // 2048, s // 2048]
	if f[0] == "splice_junctions" or f[-1] == "track_mask":
		return None
	if f[0] in ("splice_sites", "splice_site_usage"):
		return [s]
	return [s // int(f[1])]


def bounds(key):
	f = key.split(".")
	if f[-1] in ("1", "128", "pred_counts"):
		return [0, float("inf")]
	if f[-1] in ("probs", "predictions"):
		return [0, 1]
	return None


def centre(key, t, s):
	# middle 2048 bp of a per-position output
	r = rows(key, s)
	if r is None or len(r) != 1:
		return None
	k = 2048 * r[0] // s
	lo = (r[0] - k) // 2
	return t[0, lo:lo + k].float().cpu().flatten()


model = common.load_model()
print("#", json.dumps(common.env()))
fail = []
sizes = {}
mids = {}
done = []

# step 1: one forward per length, every output checked
for s in sorted(a.lengths):
	try:
		out = common.flatten(common.forward(model, common.load_input(s)))
	except torch.cuda.OutOfMemoryError:
		print(f"{s}: out of memory, skipped")
		torch.cuda.empty_cache()
		continue
	bad = []
	for k, t in out.items():
		r = rows(k, s)
		b = bounds(k)
		if r is not None and list(t.shape[1:1 + len(r)]) != r:
			bad.append(f"{k} shape {list(t.shape)}, rows should be {r}")
		if not torch.isfinite(t).all():
			bad.append(f"{k} has {(~torch.isfinite(t)).sum().item()} nan/inf")
		if b is not None and (t.min().item() < b[0] or t.max().item() > b[1]):
			bad.append(f"{k} range [{t.min().item():.3g}, {t.max().item():.3g}], should be {b}")
		c = centre(k, t, s)
		if c is not None:
			mids.setdefault(k, {})[s] = c
		sizes.setdefault(k, {})[s] = list(t.shape)
		if a.verbose:
			print(f"  {k:40s} {t.dtype} {list(t.shape)}")
	print(f"{s}: {len(out)} outputs, {'ok' if not bad else f'{len(bad)} failed'}")
	for i in bad:
		print("  fail:", i)
	fail += bad
	done.append(s)
	del out
	torch.cuda.empty_cache()

# step 2: same outputs, same track counts; fixed-size outputs identical
for k, v in sizes.items():
	first = v[done[0]]
	for s in done:
		if s not in v or v[s][-1] != first[-1] or (rows(k, s) is None and v[s] != first):
			fail.append(f"{k} differs at {s}: {v.get(s)} vs {first}")
			print("  fail:", fail[-1])
			break

# step 3: pearson r of the centre 2048 bp against the longest length
if len(done) > 1:
	top = done[-1]
	print("centre r vs", top, ":", " ".join(str(s) for s in done[:-1]))
	for k, v in mids.items():
		r = []
		for s in done[:-1]:
			r.append(f"{torch.corrcoef(torch.stack([v[s], v[top]]))[0, 1].item():.3f}")
		print(f"  {k:32s}", " ".join(r))

print("all checks passed" if not fail else f"{len(fail)} checks failed")
sys.exit(1 if fail else 0)
