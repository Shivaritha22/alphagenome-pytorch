"""cuda memory snapshot of one forward: snapshot.pickle and memory.json."""
import argparse, json
import torch
import common

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, required=True)
p.add_argument("--mode", choices=["eager", "compile_e2e"], required=True)
a = p.parse_args()

d = common.run_dir(a.length, a.mode)
model = common.load_model()
f = common.prepare(model, a.mode)
x = common.load_input(a.length)
res = {"status": "ok", "error": "", "allocated_before_bytes": 0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
rec = False
before = common.gpu_state()
try:
	for _ in range(3):
		common.forward(f, x)
	torch.cuda.synchronize()
	torch.cuda.reset_peak_memory_stats()
	res["allocated_before_bytes"] = torch.cuda.memory_allocated()
	torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python", clear_history=True, compile_context=(a.mode == "compile_e2e"))
	rec = True
	out = common.forward(f, x)
	torch.cuda.synchronize()
	torch.cuda.memory._dump_snapshot(str(d / "snapshot.pickle"), augment_with_fx_traces=True)
except Exception as e:
	if not common.oom(e):
		raise
	res["status"] = "oom"
	res["error"] = (str(e).splitlines() or [""])[0][:300]
	# dump inside except so tensors held by the traceback are still alive
	if rec:
		torch.cuda.memory._dump_snapshot(str(d / "snapshot.pickle"), augment_with_fx_traces=True)
if rec:
	torch.cuda.memory._record_memory_history(enabled=None)
res["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
res["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
json.dump(res, open(d / "memory.json", "w"), indent=1)
after = common.gpu_state()
common.manifest(d, "memory", a.length, a.mode, x, before, after, status=res["status"], error=res["error"])
