"""Smoke test: do tensor names survive quantization, and can we instrument both graphs?

Both assumptions are load-bearing for layer-by-layer analysis, so verify them
against real artifacts before building the full comparison on top.

Run after 01_export_onnx.py and 03_quantize.py.
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import onnx  # noqa: E402

from w1_quant.common import (  # noqa: E402
    intersect_tensors,
    instrument_explicit,
    extract_qparams,
    redirect_tempdir,
)

TMPDIR = redirect_tempdir()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASELINE = os.path.join(ROOT, "artifacts/models/mobilenet_v2_opset13_sim_preproc.onnx")
QUANT = os.path.join(ROOT, "artifacts/models/mobilenet_v2_opset13_sim_qdq_perchannel_minmax.onnx")
WORK = os.path.join(ROOT, "artifacts/instrumented")


def op_signature(path: str) -> tuple[list[str], list[tuple[str, str]]]:
    model = onnx.load(path)
    conv_like = [
        (n.op_type, n.output[0])
        for n in model.graph.node
        if n.op_type in ("Conv", "MatMul", "Gemm")
    ]
    tensor_names = [n.output[0] for n in model.graph.node if n.output]
    return tensor_names, conv_like


def main() -> int:
    for path in (BASELINE, QUANT):
        if not os.path.exists(path):
            print("MISSING:", path)
            print("run 01_export_onnx.py and 03_quantize.py first")
            return 1

    print("=" * 70)
    print("1. tensor name comparison between FP32 baseline and QDQ INT8")
    print("=" * 70)
    fp32_names, fp32_conv = op_signature(BASELINE)
    int8_names, int8_conv = op_signature(QUANT)
    print(f"  FP32 nodes with output : {len(fp32_names)}")
    print(f"  INT8 nodes with output : {len(int8_names)}")
    print(f"  FP32 conv-like ops     : {len(fp32_conv)}")
    print(f"  INT8 conv-like ops     : {len(int8_conv)}")

    shared = set(fp32_names) & set(int8_names)
    print(f"  shared tensor names    : {len(shared)}")

    fp32_conv_names = [name for _op, name in fp32_conv]
    int8_conv_names = [name for _op, name in int8_conv]
    conv_shared = [n for n in fp32_conv_names if n in set(int8_conv_names)]
    print(f"  shared conv outputs    : {len(conv_shared)} / {len(fp32_conv_names)}")

    same_order = fp32_conv_names == int8_conv_names
    print(f"  conv output order same : {same_order}")
    if not same_order and conv_shared:
        print("    first 5 FP32:", fp32_conv_names[:5])
        print("    first 5 INT8:", int8_conv_names[:5])

    print()
    print("=" * 70)
    print("2. quantization parameters readable from the QDQ graph")
    print("=" * 70)
    qparams = extract_qparams(QUANT)
    per_channel = [q for q in qparams if q["per_channel"]]
    print(f"  QuantizeLinear nodes   : {len(qparams)}")
    print(f"  per-channel ones       : {len(per_channel)}")
    for row in qparams[:6]:
        print(
            "    %-52s scale=%.3e..%.3e zp=%.1f ch=%d axis=%s"
            % (
                (row["tensor"] or "?")[:52],
                row["scale_min"],
                row["scale_max"],
                row["zero_point"],
                row["num_scales"],
                row["axis"],
            )
        )
    if per_channel:
        print("\n  top-5 largest per-channel scale spread (why per-channel wins):")
        for row in sorted(per_channel, key=lambda r: -r["scale_spread"])[:5]:
            print(
                "    %-52s spread=%.1fx over %d channels"
                % ((row["tensor"] or "?")[:52], row["scale_spread"], row["num_scales"])
            )

    print()
    print("=" * 70)
    print("3. instrument both graphs on the *intersection* and compare")
    print("=" * 70)
    os.makedirs(WORK, exist_ok=True)

    shared = intersect_tensors(BASELINE, [QUANT])
    print(f"  comparable tensors (intersection) : {len(shared)}")
    if not shared:
        print("  !! nothing comparable; layer analysis is impossible as written")
        return 1

    names = [layer["tensor"] for layer in shared]
    paths = {}
    for label, path in (("fp32", BASELINE), ("int8", QUANT)):
        out = os.path.join(WORK, f"{os.path.basename(path)[:-5]}_shared.onnx")
        if not os.path.exists(out):
            instrument_explicit(path, out, names)
        paths[label] = out
        print(f"  {label}: exposed {len(names)} tensors -> {os.path.basename(out)}")

    # Confirm the selection actually matters: taking only Conv outputs loses
    # the residual Adds and the classifier head.
    _, fp32_conv = op_signature(BASELINE)
    _, int8_conv = op_signature(QUANT)
    fp32_conv_names = {n for _op, n in fp32_conv}
    int8_names_all = set(int8_names)
    conv_only = [n for n in fp32_conv_names if n in int8_names_all]
    extra = [layer["tensor"] for layer in shared if layer["tensor"] not in set(conv_only)]
    from collections import Counter  # noqa: E402

    print(f"  Conv-output-only selection would give  : {len(conv_only)} layers")
    print(f"  full intersection gives               : {len(shared)} layers")
    print(f"  the {len(extra)} extra layers are          : "
          f"{dict(Counter(layer['op_type'] for layer in shared if layer['tensor'] in set(extra)))}")
    print("  -> residual Adds and the pooling head are exactly what a")
    print("     Conv-only sweep would have hidden.")

    import onnxruntime as ort

    from w1_quant.common import cosine_similarity, sqnr

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)

    s_fp32 = ort.InferenceSession(paths["fp32"], providers=["CPUExecutionProvider"])
    s_int8 = ort.InferenceSession(paths["int8"], providers=["CPUExecutionProvider"])

    names_fp32 = [o.name for o in s_fp32.get_outputs()]
    names_int8 = set(o.name for o in s_int8.get_outputs())
    print(f"  session outputs : fp32={len(names_fp32)} int8={len(names_int8)}")

    out_fp32 = dict(zip(names_fp32, s_fp32.run(None, {"input": x})))
    out_int8 = dict(zip([o.name for o in s_int8.get_outputs()],
                        s_int8.run(None, {"input": x})))

    rows = []
    for name in names_fp32:
        if name not in out_int8:
            continue
        a, b = out_fp32[name], out_int8[name]
        if a.shape != b.shape:
            continue
        rows.append(
            {"tensor": name, "cos": cosine_similarity(a, b), "sqnr_db": sqnr(a, b)}
        )
    rows.sort(key=lambda r: r["cos"])
    print(f"  compared tensors : {len(rows)}")
    print("\n  worst 8 tensors by cosine similarity:")
    print("    %-56s %10s %10s" % ("tensor", "cosine", "SQNR_dB"))
    for row in rows[:8]:
        print("    %-56s %10.6f %10.2f" % (row["tensor"][:56], row["cos"], row["sqnr_db"]))

    print(f"\n  BEST  cosine : {rows[-1]['cos']:.6f}  ({rows[-1]['tensor']})")
    print(f"  WORST cosine : {rows[0]['cos']:.6f}  ({rows[0]['tensor']})")
    print("\nSMOKE TEST PASSED - proceed to 04_layer_analysis.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
