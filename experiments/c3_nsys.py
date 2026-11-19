"""nsys timeline with nvtx module ranges of one forward for one length and mode."""
import argparse, subprocess, sys
import torch
import common

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, required=True)
p.add_argument("--mode", choices=["eager", "compile_stage"], required=True)
p.add_argument("--inner", action="store_true")
a = p.parse_args()

d = common.run_dir(a.length, a.mode)
if not a.inner:
	# re-run this script under nsys, capturing only between profiler start and stop
	cmd = ["nsys", "profile", "-t", "cuda,nvtx", "-s", "none", "--cpuctxsw=none",
		"--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true",
		"-o", str(d / "timeline"), sys.executable, __file__, "--length", str(a.length), "--mode", a.mode, "--inner"]
	sys.exit(subprocess.run(cmd).returncode)

model = common.load_model()
if a.mode == "eager":
	common.ranges(model)
else:
	model = common.prepare(model, "compile_stage")
x = common.load_input(a.length)
before = common.gpu_state()
status, error = "ok", ""
try:
	for _ in range(3):
		common.forward(model, x)
	torch.cuda.synchronize()
	torch.cuda.profiler.start()
	try:
		common.forward(model, x)
		torch.cuda.synchronize()
	finally:
		torch.cuda.profiler.stop()
except Exception as e:
	if not common.oom(e):
		raise
	status, error = "oom", str(e).splitlines()[0][:300]
after = common.gpu_state()
common.manifest(d, "nsys", a.length, a.mode, x, before, after, status=status, error=error, file="timeline.nsys-rep" if status == "ok" else "")
print(f"nsys {a.length} {a.mode} {common.prec}: {status}")
