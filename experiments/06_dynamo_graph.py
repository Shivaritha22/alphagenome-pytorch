"""Capture the model with torch.compile's front end and draw its dependency graph.

Capture only (ops, order, who-feeds-whom, owning module); nothing is fused or timed here.
Fusion is Inductor's job and is not covered. Runs the capture once, outside any timing.

Two levels (--level):
  dynamo  Python-level ops as TorchDynamo sees them (F.linear, Tensor.view, ...)
  aten    core-ATen primitives: the AOTAutograd inference graph after PrimTorch decompositions,
          i.e. what a compiler backend receives. Closer to what actually runs on the GPU.
          Metadata-only ops (view, permute, slice, ...) launch no kernel and are drawn grey/dashed.

  06_dynamo_graph.py --length 16384 [--level aten]      capture + module-level overview
  06_dynamo_graph.py --render-only --root tower.blocks.0 --depth 2 --ops
                                                        zoom: op-level graph of one module,
                                                        from the saved json (no model needed)

Outputs (OUT_DIR), <lv> = dynamo|aten, <tag> = 16kb etc:
  <lv>_<tag>.json              the captured graph + graph-break report (input to every view)
  <lv>_<tag>_<view>.dot        Graphviz source
  <lv>_<tag>_<view>.html       self-contained viewer (renders the dot in the browser, needs internet
                               for the viz-js library); .svg too if `dot` is on PATH

Views: nodes are the modules `--depth` levels below `--root` (ModuleList/ModuleDict are not
levels, matching 03_trace_tree.py); an edge A -> B means a tensor produced in A is consumed in B.
With --ops each op is its own node, clustered by module; producers/consumers outside the root
appear as grey stubs.
"""
import argparse
import html
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict

import torch

import common

VIZ_JS = "https://cdn.jsdelivr.net/npm/@viz-js/viz@3.11.0/lib/viz-standalone.js"
PALETTE = ["#cfe8ff", "#ffe3c2", "#d9f2d0", "#f6d5f0", "#fff3b0", "#d6dcff", "#ffd6d6", "#cdeee9"]
CONTAINERS = ("ModuleList", "ModuleDict")


# ---- capture ---------------------------------------------------------------

def norm_path(p: str) -> str:
    """Dynamo module paths ("L['self'].encoder", "..._modules['x']", "blocks[0]['mha']") -> dotted names."""
    p = re.sub(r"^L\['self'\]", "", p)
    p = re.sub(r"\._modules\['([^']+)'\]", r".\1", p)
    p = re.sub(r"\[(\d+)\]", r".\1", p)
    p = re.sub(r"\['([^']+)'\]", r".\1", p)  # ModuleDict access: blocks.0['mha']
    p = re.sub(r"^self\.?", "", p)
    return p.lstrip(".")


def op_label(n, gm) -> str:
    if n.op == "call_module":
        try:
            return type(gm.get_submodule(n.target)).__name__
        except Exception:
            return str(n.target)
    if n.op == "call_method":
        return "Tensor." + str(n.target)
    if n.op == "call_function":
        t = n.target
        if hasattr(t, "_overloadpacket"):  # aten op: aten.addmm.default -> aten.addmm
            return "aten." + t._overloadpacket.__name__
        name = getattr(t, "__name__", str(t))
        mod = getattr(t, "__module__", "") or ""
        if mod.endswith("functional"):
            return "F." + name
        if mod in ("_operator", "operator"):
            return name
        return "torch." + name
    return n.op


# Metadata-only ops: they re-describe a tensor (strides/shape) and launch no GPU kernel.
VIEW_OPS = {"view", "permute", "unsqueeze", "squeeze", "slice", "select", "expand", "t", "transpose",
            "alias", "as_strided", "detach", "getitem", "unflatten", "movedim", "contiguous_no_copy"}


def is_view(label: str) -> bool:
    return label.rsplit(".", 1)[-1] in VIEW_OPS


def capture_dynamo(model, x, organism):
    """Python-level ops (F.linear, Tensor.view, ...) as Dynamo sees them."""
    import torch._dynamo as dynamo

    dynamo.reset()
    with torch.no_grad():
        ex = dynamo.explain(model)(x, organism)
    breaks = [{"reason": str(r.reason)[:400],
               "where": str(r.user_stack[-1]) if getattr(r, "user_stack", None) else None}
              for r in ex.break_reasons]
    return list(ex.graphs), breaks, 0


def capture_aten(model, x, organism):
    """Core-ATen ops: the AOTAutograd forward graph after PrimTorch decompositions.
    Parameters/buffers are lifted to leading placeholders; the last two are the real inputs."""
    import torch._dynamo as dynamo
    from torch._decomp import core_aten_decompositions
    from torch._dynamo.backends.common import aot_autograd

    graphs = []

    def fw(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    backend = aot_autograd(fw_compiler=fw, decompositions=core_aten_decompositions())
    dynamo.reset()
    with torch.no_grad():
        torch.compile(model, backend=backend)(x, organism)
    return graphs, [], 2


def capture(length, level):
    device = common.default_device()
    model = common.load_model(device)
    x = common.make_input(length, device)
    organism = torch.zeros(1, dtype=torch.long, device=device)

    graphs, breaks, n_user_inputs = (capture_dynamo if level == "dynamo" else capture_aten)(model, x, organism)

    mods = {n: {"cls": type(m).__name__, "container": type(m).__name__ in CONTAINERS,
                "params": sum(p.numel() for p in m.parameters())}
            for n, m in model.named_modules()}

    nodes, ids, unresolved, unattributed = [], {}, 0, 0
    for gi, gm in enumerate(graphs):
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        lifted = set(placeholders[: len(placeholders) - n_user_inputs]) if n_user_inputs else set()
        for n in gm.graph.nodes:
            if n.op == "get_attr" or n in lifted:  # parameters/buffers: not data flow we draw
                continue
            if n.op == "placeholder":
                path, label = "<input>", f"input {n.target}"
            elif n.op == "output":
                path, label = "<output>", "output"
            else:
                stack = n.meta.get("nn_module_stack") or {}
                path = norm_path(list(stack.values())[-1][0]) if stack else ""
                if path and path not in mods:
                    unresolved += 1
                unattributed += not path
                label = op_label(n, gm)
            val = n.meta.get("example_value", n.meta.get("val"))
            ids[(gi, n)] = len(nodes)
            nodes.append({"id": len(nodes), "g": gi, "kind": n.op, "label": label, "module": path,
                          "shape": common.desc(val), "in": [], "_n": n})
    for node in nodes:
        n, gi = node.pop("_n"), node["g"]
        node["in"] = sorted({ids[(gi, i)] for i in n.all_input_nodes if (gi, i) in ids})

    n_ops = sum(1 for n in nodes if n["kind"].startswith("call"))
    data = {
        "meta": {"length": length, "device": device, "level": level, **common.env_info()},
        "summary": {"graph_count": len(graphs), "graph_break_count": len(breaks), "ops": n_ops,
                    "view_only_ops": sum(1 for n in nodes if n["kind"].startswith("call") and is_view(n["label"])),
                    "nodes": len(nodes), "unresolved_module_paths": unresolved, "ops_without_module": unattributed},
        "breaks": breaks, "modules": mods, "nodes": nodes,
    }
    path = common.OUT_DIR / f"{level}_{common.tag(length)}.json"
    common.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    s = data["summary"]
    print(f"captured [{level}]: {s['graph_count']} graph(s), {s['graph_break_count']} graph break(s), "
          f"{s['ops']} ops ({s['view_only_ops']} view-only), {s['unresolved_module_paths']} unresolved module paths, "
          f"{s['ops_without_module']} ops without a module")
    for b in breaks:
        print(f"  BREAK: {b['reason']}  @ {b['where']}")
    print(f"saved {path}")
    return data


# ---- views -------------------------------------------------------------------

def level_of(prefix, mods):
    """Number of real (non-container) modules on the way to `prefix`, inclusive."""
    segs, lvl = prefix.split("."), 0
    for i in range(len(segs)):
        m = mods.get(".".join(segs[: i + 1]))
        if m and not m["container"]:
            lvl += 1
    return lvl


def group_of(path, root, depth, mods):
    if path.startswith("<") or path == "":
        return path or "<glue>"
    target = (level_of(root, mods) if root else 0) + depth
    segs = path.split(".")
    for i in range(len(segs)):
        pre = ".".join(segs[: i + 1])
        m = mods.get(pre)
        if m and not m["container"] and level_of(pre, mods) == target:
            return pre
    return path


def in_root(path, root):
    return not root or path == root or path.startswith(root + ".")


def fmt_n(n):
    return f"{n / 1e6:.2f}M" if n >= 1e5 else f"{n / 1e3:.1f}K" if n >= 1e3 else str(n)


def shape_str(s):
    if s is None:
        return ""
    if isinstance(s, str):
        return s.replace("float32", "f32").replace("float16", "f16").replace("bfloat16", "bf16").replace("int64", "i64")
    if isinstance(s, list):
        return "(" + ", ".join(filter(None, (shape_str(x) for x in s[:3]))) + (", ..." if len(s) > 3 else "") + ")"
    if isinstance(s, dict):
        return "{" + ", ".join(f"{k}:{shape_str(v)}" for k, v in list(s.items())[:3]) + "}"
    return ""


def q(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def stage_colors():
    sm = common.load_stage_map()
    names = list(sm)

    def color(path):
        best, blen = None, -1
        for st, prefixes in sm.items():
            for p in prefixes:
                if (path == p or path.startswith(p + ".")) and len(p) > blen:
                    best, blen = st, len(p)
        return PALETTE[names.index(best) % len(PALETTE)] if best else "#eeeeee"

    return color, sm


def is_side_input(node):
    """Integer placeholders (the organism index) feed every head; drawing them buries the real flow."""
    return node["kind"] == "placeholder" and shape_str(node["shape"]).startswith(("i64", "int"))


def view_overview(data, root, depth):
    mods, color = data["modules"], stage_colors()[0]
    sel = [n for n in data["nodes"] if in_root(n["module"], root) or n["module"].startswith("<")]
    grp = {n["id"]: group_of(n["module"], root, depth, mods) for n in sel}
    ops = Counter(grp[n["id"]] for n in sel if n["kind"].startswith("call"))
    views = Counter(grp[n["id"]] for n in sel if n["kind"].startswith("call") and is_view(n["label"]))
    edges = defaultdict(Counter)
    for n in sel:
        for i in n["in"]:
            if is_side_input(data["nodes"][i]):
                continue
            if i in grp and grp[i] != grp[n["id"]]:
                edges[(grp[i], grp[n["id"]])][shape_str(data["nodes"][i]["shape"])] += 1
    names = sorted({g for e in edges for g in e} | set(ops), key=lambda g: min((n["id"] for n in sel if grp[n["id"]] == g), default=0))
    lines = ["digraph G {", '  rankdir=TB; node [shape=box, style="rounded,filled", fontname=Helvetica, fontsize=11]; edge [fontname=Helvetica, fontsize=9, color="#7a3fa0"];']
    for g in names:
        m = mods.get(g)
        parts = [g]
        if m:
            parts.append(f"{m['cls']}  {fmt_n(m['params'])} params")
        if ops[g]:
            parts.append(f"{ops[g]} ops ({views[g]} view-only)")
        label = "\n".join(parts)  # q() turns the newline into DOT's \n
        fill = color(g) if m else "#ffffff"
        lines.append(f"  {q(g)} [label={q(label)}, fillcolor={q(fill)}];")
    for (a, b), shapes in edges.items():
        top = ", ".join(s for s, _ in shapes.most_common(2) if s)
        lines.append(f"  {q(a)} -> {q(b)} [label={q(f'{sum(shapes.values())}x {top}')}];")
    lines.append("}")
    return "\n".join(lines), len(names), len(edges)


def view_ops(data, root, depth, max_nodes):
    mods, color = data["modules"], stage_colors()[0]
    nodes = data["nodes"]
    sel = [n for n in nodes if n["kind"].startswith("call") and in_root(n["module"], root)]
    if not sel:
        raise SystemExit(f"no ops under root '{root}'")
    if len(sel) > max_nodes:
        raise SystemExit(f"{len(sel)} ops under '{root}' > --max-nodes {max_nodes}; pick a deeper --root or raise --max-nodes")
    selected = {n["id"] for n in sel}
    clusters = defaultdict(list)
    for n in sel:
        clusters[group_of(n["module"], root, depth, mods)].append(n)
    lines = ["digraph G {", '  rankdir=TB; compound=true; node [shape=box, style="rounded,filled", fillcolor="#ffe3c2", fontname=Helvetica, fontsize=10]; edge [color="#7a3fa0", arrowsize=0.7];']
    for ci, (g, ns) in enumerate(clusters.items()):
        m = mods.get(g)
        lines.append(f'  subgraph cluster_{ci} {{ label={q(g + (" [" + m["cls"] + "]" if m else ""))}; color="#d62728"; fontname=Helvetica; style=rounded;')
        for n in ns:
            style = ', style="rounded,dashed,filled", fillcolor="#f0f0f0"' if is_view(n["label"]) else ""  # no kernel
            lines.append(f"    n{n['id']} [label={q(n['label'] + chr(10) + shape_str(n['shape']))}{style}];")
        lines.append("  }")
    stubs = {}
    for n in sel:
        for i in n["in"]:
            if i in selected:
                lines.append(f"  n{i} -> n{n['id']};")
            else:
                src = nodes[i]
                g = src["module"] if src["module"].startswith("<") else group_of(src["module"], "", depth, mods) or "<glue>"
                stubs.setdefault(g, f"s{len(stubs)}")
                lines.append(f"  {stubs[g]} -> n{n['id']} [style=dashed, label={q(shape_str(src['shape']))}, fontsize=8];")
    consumers = defaultdict(set)
    for n in nodes:
        if n["id"] not in selected:
            for i in n["in"]:
                if i in selected:
                    consumers[group_of(n["module"], "", depth, mods) if not n["module"].startswith("<") else n["module"]].add(i)
    for g, srcs in consumers.items():
        key = "c_" + g
        stubs.setdefault(key, f"s{len(stubs)}")
        for i in sorted(srcs):
            lines.append(f"  n{i} -> {stubs[key]} [style=dashed];")
    for g, sid in stubs.items():
        lines.append(f"  {sid} [label={q(g.replace('c_', 'to: ', 1) if g.startswith('c_') else 'from: ' + g)}, shape=box, style=\"dashed,filled\", fillcolor=\"#eeeeee\"];")
    lines.append("}")
    return "\n".join(lines), len(sel), len(clusters)


HTML = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{margin:0;font-family:Helvetica,Arial,sans-serif}}#bar{{position:sticky;top:0;background:#222;color:#fff;padding:6px 10px;z-index:1}}
#bar button{{margin-left:6px}}#g{{padding:10px;overflow:auto}}</style>
<div id="bar">{title} &nbsp; <button onclick="z(1.25)">+</button><button onclick="z(0.8)">-</button><button onclick="fit()">fit</button> <span id="msg"></span></div>
<div id="g">rendering...</div>
<script type="text/plain" id="dot">{dot}</script>
<script src="{viz}"></script>
<script>
let s=1,svg;
function z(f){{s*=f;svg.style.width=(svg.dataset.w*s)+'px';svg.style.height='auto';}}
function fit(){{s=Math.min(1.5,(window.innerWidth-30)/svg.dataset.w);z(1);}}
Viz.instance().then(v=>{{svg=v.renderSVGElement(document.getElementById('dot').textContent);
 svg.dataset.w=parseFloat(svg.getAttribute('width'));document.getElementById('g').replaceChildren(svg);fit();}})
 .catch(e=>{{document.getElementById('g').textContent='render failed: '+e+' (needs internet for viz-js; the .dot file has the graph)';}});
</script>"""


def write_view(dot, base, title):
    common.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (common.OUT_DIR / f"{base}.dot").write_text(dot)
    (common.OUT_DIR / f"{base}.html").write_text(HTML.format(title=html.escape(title), dot=dot.replace("</", "<\\/"), viz=VIZ_JS))
    out = [f"{base}.dot", f"{base}.html"]
    if shutil.which("dot"):
        subprocess.run(["dot", "-Tsvg", str(common.OUT_DIR / f"{base}.dot"), "-o", str(common.OUT_DIR / f"{base}.svg")], check=True)
        out.append(f"{base}.svg")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--root", default="", help="module name prefix to draw (default: whole model)")
    ap.add_argument("--depth", type=int, default=1, help="module levels below --root that become nodes/clusters")
    ap.add_argument("--ops", action="store_true", help="op-level graph of --root (one node per op)")
    ap.add_argument("--max-nodes", type=int, default=400)
    ap.add_argument("--level", choices=["dynamo", "aten"], default="dynamo",
                    help="dynamo: Python-level ops; aten: core-ATen primitives (AOTAutograd forward + PrimTorch decomps)")
    ap.add_argument("--render-only", action="store_true", help="reuse the saved json; no model, no capture")
    args = ap.parse_args()

    t, lv = common.tag(args.length), args.level
    data = json.load(open(common.OUT_DIR / f"{lv}_{t}.json")) if args.render_only else capture(args.length, lv)

    slug = (args.root or "all").replace(".", "-")
    if args.ops:
        dot, n_nodes, n_clusters = view_ops(data, args.root, args.depth, args.max_nodes)
        base = f"{lv}_{t}_{slug}_d{args.depth}_ops"
        title = f"[{lv}] {args.root or 'model'}: {n_nodes} ops in {n_clusters} modules ({t}); grey dashed = metadata-only (no kernel)"
    else:
        dot, n_nodes, _ = view_overview(data, args.root, args.depth)
        base = f"{lv}_{t}_{slug}_d{args.depth}"
        title = f"[{lv}] {args.root or 'model'}: {n_nodes} module nodes ({t}); integer inputs (organism index) not drawn"
    files = write_view(dot, base, title)
    print(f"view: {title}")
    for f in files:
        print(f"saved {common.OUT_DIR / f}")


if __name__ == "__main__":
    main()
