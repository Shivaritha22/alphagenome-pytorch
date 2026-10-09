"""Trace one forward pass into a full-depth JSON tree (structure only, NO timing).

Forward hooks on every module give the nesting through a call stack. A
TorchFunctionMode catches every torch.* / F.* / Tensor-method call and attaches it
as a leaf op under whichever module is currently on top of the stack. Ops made by
model.forward() itself (glue) therefore land directly under the root.

Hooks and the mode are removed afterwards; the model code is untouched.
Never time with this on; use the harness for numbers.
"""
import argparse
import json

import torch
from torch.overrides import TorchFunctionMode

import common

# Noise: property getters, metadata queries, python protocol methods.
SKIP = {
    "__get__", "__set__", "__len__", "__repr__", "__format__", "__hash__", "__bool__",
    "__index__", "__iter__", "__torch_function__", "size", "dim", "numel", "ndimension",
    "nelement", "stride", "storage_offset", "element_size", "data_ptr", "get_device",
    "is_floating_point", "is_complex", "is_contiguous", "type", "__reduce_ex__",
}


def op_name(func) -> str:
    name = getattr(func, "__name__", str(func))
    qual = getattr(func, "__qualname__", name)
    mod = getattr(func, "__module__", "") or ""
    if "Tensor" in qual:
        return "Tensor." + name
    if mod.endswith("functional"):
        return "F." + name
    return "torch." + name


def tensor_args(args, kwargs, limit=4):
    out = []
    for o in list(args) + list(kwargs.values()):
        if torch.is_tensor(o):
            out.append(common.desc(o))
        elif isinstance(o, (list, tuple)):
            out += [common.desc(t) for t in o if torch.is_tensor(t)]
        if len(out) >= limit:
            break
    return out[:limit]


class OpTracer(TorchFunctionMode):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if getattr(func, "__name__", "") not in SKIP:
            node = {"type": "op", "op": op_name(func), "in": tensor_args(args, kwargs)}
            o = common.desc(out)
            if o is not None:
                node["out"] = o
            self.stack[-1]["children"].append(node)
        return out


def attach_hooks(model, stack, calls):
    params = {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_modules()}
    handles = []
    for name, mod in model.named_modules():
        def pre(m, args, kwargs, name=name):
            calls[name] = calls.get(name, 0) + 1
            node = {
                "type": "module", "name": name, "cls": type(m).__name__, "call": calls[name],
                "params": params[name], "in": common.desc((args, kwargs)), "children": [],
            }
            stack[-1]["children"].append(node)
            stack.append(node)

        def post(m, args, kwargs, out):
            stack.pop()["out"] = common.desc(out)

        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(mod.register_forward_hook(post, with_kwargs=True))
    return handles


def sanity(model, x, length):
    """Untraced run: confirm output keys/shapes are sane and finite."""
    out = common.run_forward(model, x)
    shapes = {k: common._t(t) for k, t in common.iter_tensors(out)}
    bad = [k for k, t in common.iter_tensors(out) if not torch.isfinite(t).all()]
    print(f"sanity {common.tag(length)}: {len(shapes)} output tensors, non-finite: {bad or 'none'}")
    for k, s in shapes.items():
        print(f"  {k:45s} {s}")
    if bad:
        raise RuntimeError(f"non-finite outputs: {bad}")
    return shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=16384)
    args = ap.parse_args()

    device = common.default_device()
    model = common.load_model(device)
    x = common.make_input(args.length, device)
    shapes = sanity(model, x, args.length)

    top = {"type": "module", "name": "<top>", "children": []}
    stack, calls = [top], {}
    handles = attach_hooks(model, stack, calls)
    try:
        with OpTracer(stack):
            common.run_forward(model, x)
    finally:
        for h in handles:
            h.remove()
    assert len(stack) == 1, "hook stack not balanced"

    # run_forward's own torch.zeros (organism index) lands beside the model call; not model work
    root = next(c for c in top["children"] if c["type"] == "module" and c["name"] == "")
    result = {
        "meta": {"length": args.length, "device": device, "dtype": "fp32", **common.env_info()},
        "outputs": shapes,
        "calls": calls,
        "top_level_order": [c["name"] for c in root["children"] if c["type"] == "module"],
        "tree": root,
    }
    common.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = common.OUT_DIR / f"tree_{common.tag(args.length)}.json"
    path.write_text(json.dumps(result))
    n_ops = sum(1 for _ in _walk(root) if _["type"] == "op")
    print(f"saved {path}  ({n_ops} ops, {len(calls)} modules, {path.stat().st_size / 1e6:.1f} MB)")


def _walk(node):
    yield node
    for c in node.get("children", []):
        yield from _walk(c)


if __name__ == "__main__":
    main()
