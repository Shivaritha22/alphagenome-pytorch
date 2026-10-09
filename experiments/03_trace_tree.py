"""Print the traced tree as readable text; check/annotate the stage map.

  03_trace_tree.py --depth 2                       top level
  03_trace_tree.py --root tower.blocks.0 --depth 3 --ops
  03_trace_tree.py --stages                        per-stage table + coverage + pair_update check
  03_trace_tree.py --full --ops --save         write tree_<tag>.txt (stage-tagged)

--root takes a name prefix. tower.blocks.N is a ModuleDict that is never *called*, so a
prefix selects the called modules beneath it (mha, mlp, pair_update, ...).
"""
import argparse
import json
import re
from itertools import groupby

import common


def walk(node):
    yield node
    for c in node.get("children", []):
        yield from walk(c)


def fmt_n(n):
    return f"{n / 1e6:.2f}M" if n >= 1e5 else f"{n / 1e3:.1f}K" if n >= 1e3 else str(n)


def fmt_io(x, width=70):
    s = json.dumps(x, separators=(",", ":")).replace('"', "") if x is not None else "-"
    return s if len(s) <= width else s[: width - 3] + "..."


def stage_of(name, stage_map):
    """Longest-prefix stage match; '' (the root) is glue."""
    best, best_len = None, -1
    for stage, prefixes in stage_map.items():
        for p in prefixes:
            if (name == p or name.startswith(p + ".")) and len(p) > best_len:
                best, best_len = stage, len(p)
    return best


def matches(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def select_roots(tree, root):
    if not root:
        return [tree]
    hits = []

    def rec(node):
        for c in node.get("children", []):
            if c["type"] == "module" and matches(c["name"], root):
                hits.append(c)
            else:
                rec(c)

    rec(tree)
    # keep only the topmost matches (a matched node's descendants are shown under it)
    return hits


def render(node, calls, stage_map, depth, show_ops, lvl=0, lines=None):
    lines = [] if lines is None else lines
    pad = "  " * lvl
    if node["type"] == "module":
        n = calls.get(node["name"], 1)
        call = f" call {node['call']}/{n}" if n > 1 else ""
        st = stage_of(node["name"], stage_map) if stage_map else None
        tag = f"  [{st or ('glue' if node['name'] == '' else '?')}]" if stage_map else ""
        lines.append(
            f"{pad}{node['name'] or '<root>'} [{node['cls']}]{call}  params={fmt_n(node['params'])}"
            f"  in={fmt_io(node.get('in'))} -> out={fmt_io(node.get('out'))}{tag}"
        )
        if lvl >= depth:
            return lines
    kids = node.get("children", [])
    if show_ops:
        for is_op, grp in groupby(kids, key=lambda c: c["type"] == "op"):
            grp = list(grp)
            if is_op:
                for op, same in groupby(grp, key=lambda c: c["op"]):
                    same = list(same)
                    x = same[0]
                    cnt = f" x{len(same)}" if len(same) > 1 else ""
                    lines.append(f"{pad}  . {op}{cnt}  {fmt_io(x.get('in'), 60)} -> {fmt_io(x.get('out'), 40)}")
            else:
                for c in grp:
                    render(c, calls, stage_map, depth, show_ops, lvl + 1, lines)
    else:
        for c in kids:
            if c["type"] == "module":
                render(c, calls, stage_map, depth, show_ops, lvl + 1, lines)
    return lines


def stage_report(data, stage_map):
    tree, calls = data["tree"], data["calls"]
    mods = [n for n in walk(tree) if n["type"] == "module"]
    by_name = {}
    for m in mods:
        by_name.setdefault(m["name"], []).append(m)

    # A prefix may be an uncalled container (heads, tower.blocks.0): aggregate the called
    # modules beneath it. calls = max calls of any member; params = distinct members summed.
    out = ["", "STAGE TABLE", f"{'stage':22s}{'prefix':36s} {'calls':>5s} {'params':>9s}  in -> out"]
    for stage, prefixes in stage_map.items():
        for p in prefixes:
            hits = select_roots(tree, p)
            if not hits:
                out.append(f"{stage:22s} {p:36s} {'-':>5s} {'-':>9s}  NOT CALLED in trace (check prefix)")
                continue
            distinct = {h["name"]: h for h in hits}
            n_calls = max(len(by_name[n]) for n in distinct)
            params = sum(h["params"] for h in distinct.values())
            label = p if len(distinct) == 1 else f"{p} (+{len(distinct)} modules)"
            out.append(
                f"{stage:22s} {label:36s} {n_calls:5d} {fmt_n(params):>9s}  "
                f"{fmt_io(hits[0].get('in'), 45)} -> {fmt_io(hits[-1].get('out'), 45)}"
            )

    top = [c for c in tree["children"] if c["type"] == "module"]
    unc = [c["name"] for c in top if stage_of(c["name"], stage_map) is None]
    out.append("")
    out.append(f"top-level call order: {data['top_level_order']}")
    out.append(f"uncovered top-level modules: {unc or 'none'}")

    pu = sorted(int(m.group(1)) for n in by_name if (m := re.fullmatch(r"tower\.blocks\.(\d+)\.pair_update", n)))
    out.append(f"pair_update called in blocks: {pu}  -> even-only: {all(i % 2 == 0 for i in pu)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--json", help="override tree json path")
    ap.add_argument("--depth", type=int, default=2, help="module levels below the root shown")
    ap.add_argument("--full", action="store_true", help="no depth limit (overrides --depth)")
    ap.add_argument("--root", default="", help="module name prefix to start from")
    ap.add_argument("--ops", action="store_true", help="show leaf torch ops (consecutive repeats collapsed)")
    ap.add_argument("--stages", action="store_true", help="per-stage table, coverage, pair_update check")
    ap.add_argument("--save", action="store_true", help="write tree_<tag>.txt next to the json")
    args = ap.parse_args()

    depth = float("inf") if args.full else args.depth
    path = args.json or common.OUT_DIR / f"tree_{common.tag(args.length)}.json"
    data = json.load(open(path))
    stage_map = common.load_stage_map()

    lines = [f"# {path.name if hasattr(path, 'name') else path}  meta={json.dumps(data['meta'])}"]
    for r in select_roots(data["tree"], args.root):
        lines += render(r, data["calls"], stage_map, depth, args.ops)
    if args.stages:
        if not stage_map:
            raise SystemExit("no stage_map.json next to the scripts")
        lines += stage_report(data, stage_map)

    text = "\n".join(lines)
    print(text)
    if args.save:
        out = common.OUT_DIR / f"tree_{common.tag(data['meta']['length'])}.txt"
        out.write_text(text + "\n")
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
