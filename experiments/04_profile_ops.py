"""Per-op / per-kernel view with torch.profiler (times are inflated; use the harness for numbers).

Outputs (in OUT_DIR): profile_<tag>.txt (op table) and perfetto_<tag>.json (open in ui.perfetto.dev).
If stage_map.json exists, each stage prefix is wrapped in a record_function range so the
timeline shows stage boundaries.
"""
import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import common


def add_stage_ranges(model, stage_map):
    handles, open_ctx = [], {}
    names = dict(model.named_modules())
    for stage, prefixes in stage_map.items():
        for p in prefixes:
            mod = names.get(p)
            if mod is None:
                continue

            def pre(m, a, label=f"{stage}:{p}", key=p):
                ctx = record_function(label)
                ctx.__enter__()
                open_ctx.setdefault(key, []).append(ctx)

            def post(m, a, o, key=p):
                open_ctx[key].pop().__exit__(None, None, None)

            handles += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    device = common.default_device()
    model = common.load_model(device)
    x = common.make_input(args.length, device)
    for _ in range(args.warmup):
        common.run_forward(model, x)
    if device == "cuda":
        torch.cuda.synchronize()

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    handles = add_stage_ranges(model, common.load_stage_map())
    try:
        with profile(activities=acts, record_shapes=True) as prof:
            common.run_forward(model, x)
            if device == "cuda":
                torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()

    sort = None
    for cand in (["self_device_time_total", "self_cuda_time_total"] if device == "cuda" else []) + ["self_cpu_time_total"]:
        try:
            table = prof.key_averages().table(sort_by=cand, row_limit=args.rows)
            sort = cand
            break
        except Exception:
            continue

    t = common.tag(args.length)
    common.OUT_DIR.mkdir(parents=True, exist_ok=True)
    txt = common.OUT_DIR / f"profile_{t}.txt"
    txt.write_text(f"# {common.env_info()}\n# sorted by {sort}; profiler overhead inflates times\n\n{table}\n")
    trace = common.OUT_DIR / f"perfetto_{t}.json"
    prof.export_chrome_trace(str(trace))
    print(table)
    print(f"\nsaved {txt}\nsaved {trace}  (open at ui.perfetto.dev)")


if __name__ == "__main__":
    main()
