"""Carry results from the Colab VM back to your machine through the notebook itself.

  Colab : python experiments/export_results.py export
          zips experiments/results/ and prints it as base64 between two marker lines.
  Local : python experiments/export_results.py unpack experiments/run.ipynb
          reads the saved notebook's outputs, finds the last export block, and extracts it
          into experiments/results/.

Needs no Drive and no browser download: the notebook file saved by the editor is the transport.
"""
import argparse
import base64
import io
import json
import zipfile

import common

BEGIN = "=====BEGIN_AG_RESULTS_ZIP====="
END = "=====END_AG_RESULTS_ZIP====="


def export():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        files = sorted(p for p in common.OUT_DIR.rglob("*") if p.is_file())
        for p in files:
            z.write(p, p.relative_to(common.OUT_DIR).as_posix())
    b64 = base64.b64encode(buf.getvalue()).decode()
    print(f"{len(files)} files, zip {len(buf.getvalue()) / 1e6:.2f} MB, base64 {len(b64) / 1e6:.2f} MB")
    for p in files:
        print(f"  {p.relative_to(common.OUT_DIR).as_posix()}  ({p.stat().st_size / 1e3:.0f} KB)")
    print(BEGIN)
    for i in range(0, len(b64), 120):
        print(b64[i:i + 120])
    print(END)


def unpack(notebook):
    nb = json.load(open(notebook, encoding="utf-8"))
    text = "".join(
        "".join(o.get("text", [])) for c in nb["cells"] for o in c.get("outputs", []) if o.get("output_type") == "stream"
    )
    if BEGIN not in text or END not in text:
        raise SystemExit("no export block in the notebook outputs: run the export cell, then save the notebook")
    block = text[text.rindex(BEGIN) + len(BEGIN):text.rindex(END)]
    data = base64.b64decode("".join(block.split()))
    common.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(common.OUT_DIR)
        for n in z.namelist():
            print(f"extracted {common.OUT_DIR / n}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    up = sub.add_parser("unpack")
    up.add_argument("notebook")
    args = ap.parse_args()
    export() if args.cmd == "export" else unpack(args.notebook)


if __name__ == "__main__":
    main()
