"""W1-D6 experiment 1: per-layer quantization error and "which layer breaks first".

Three questions this answers with real numbers:

1. How much does each layer's output drift after INT8 quantization?
2. Which layer is the **first** to collapse (the "第一个崩掉的层")?
3. Does the error grow monotonically, or stay flat until a specific op?

Method
------
Both the FP32 baseline and every INT8 variant are instrumented so their
intermediate Conv/Add/etc. outputs become graph outputs (``instrument_graph``).
Because ONNX Runtime's QDQ quantizer keeps the original op nodes and only wraps
their inputs with QuantizeLinear/DequantizeLinear, tensor names are shared
between the two graphs, so outputs can be matched by name.

For each comparable tensor we report:

* ``cosine``    - direction preservation; < 0.99 is a real distortion
* ``sqnr_db``   - signal-to-quantization-noise ratio in dB
* ``rel_l2``    - relative L2 error
* ``input_quant_err`` - how much the *input* to that layer was distorted by
  QuantizeLinear alone. Comparing this with the layer's own output error tells
  you whether the damage is local or inherited.

Usage
-----
    python 04_layer_analysis.py --calib data/calib/calib_all.txt --num-images 32
    python 04_layer_analysis.py --variants qdq_pertensor_minmax qdq_perchannel_minmax
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w1_quant.common import (  # noqa: E402
    cosine_similarity,
    extract_qparams,
    format_table,
    instrument_explicit,
    intersect_tensors,
    load_layer_meta,
    redirect_tempdir,
    rel_l2,
    rms,
    sqnr,
    write_csv,
    write_json,
)

TMPDIR = redirect_tempdir()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG")


def resolve_images(spec: str, limit: int) -> list[str]:
    if os.path.isdir(spec):
        paths: list[str] = []
        for ext in IMAGE_EXTS:
            paths.extend(glob.glob(os.path.join(spec, "**", ext), recursive=True))
        paths = sorted(set(paths))
    else:
        with open(spec, encoding="utf-8") as fh:
            paths = [line.strip() for line in fh if line.strip()]
    return paths[:limit]


def load_images(paths: list[str], size: int = 224):
    import cv2

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    out = []
    for path in paths:
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            continue
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        x = (img.astype(np.float32) / 255.0 - mean) / std
        out.append(np.transpose(x, (2, 0, 1)))
    return out


def instrument_cached(model_path: str, work_dir: str, tensors: list[str]) -> tuple[str, dict]:
    os.makedirs(work_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    out = os.path.join(work_dir, f"{stem}_shared.onnx")
    if not os.path.exists(out):
        meta = instrument_explicit(model_path, out, tensors)
    else:
        meta = load_layer_meta(out)
    return out, meta


# --------------------------------------------------------------------------
# batched activation comparison
# --------------------------------------------------------------------------


def compare_models(
    fp32_instr: str,
    int8_instr: str,
    images: list[np.ndarray],
    batch_size: int = 4,
    chunk: int = 24,
) -> dict[str, dict]:
    """Run both graphs and accumulate per-tensor metrics over all images.

    Tensors are processed in chunks because exposing every intermediate tensor at
    once makes a single session allocate hundreds of MB.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s_fp32 = ort.InferenceSession(fp32_instr, opts, providers=["CPUExecutionProvider"])
    s_int8 = ort.InferenceSession(int8_instr, opts, providers=["CPUExecutionProvider"])

    fp32_names = [o.name for o in s_fp32.get_outputs()]
    int8_names = set(o.name for o in s_int8.get_outputs())
    shared = [n for n in fp32_names if n in int8_names]
    if not shared:
        raise RuntimeError("no shared tensors between the two graphs")

    meta = {o.name: o for o in s_fp32.get_outputs()}
    input_name = s_fp32.get_inputs()[0].name

    # The exported graph has a fixed batch dimension, so honour it instead of
    # assuming the requested batch size is usable.
    fixed = s_fp32.get_inputs()[0].shape[0]
    if isinstance(fixed, int) and fixed > 0 and batch_size != fixed:
        print(f"  note: model input batch is fixed at {fixed}; "
              f"overriding batch-size {batch_size} -> {fixed}")
        batch_size = fixed

    accum: dict[str, dict] = {
        name: {"cos": [], "sqnr": [], "rel": [], "absmax": [], "rms": []}
        for name in shared
    }

    for start in range(0, len(images), batch_size):
        batch = np.stack(images[start : start + batch_size], axis=0)
        for offset in range(0, len(shared), chunk):
            group = shared[offset : offset + chunk]
            outs_fp32 = s_fp32.run(group, {input_name: batch})
            outs_int8 = s_int8.run(group, {input_name: batch})
            for name, a, b in zip(group, outs_fp32, outs_int8):
                if a.shape != b.shape:
                    continue
                rec = accum[name]
                rec["cos"].append(cosine_similarity(a, b))
                rec["sqnr"].append(sqnr(a, b))
                rec["rel"].append(rel_l2(a, b))
                rec["absmax"].append(float(np.max(np.abs(a))))
                rec["rms"].append(rms(a))
        print(
            "  batch %d/%d done"
            % (
                min(start + batch_size, len(images)) // batch_size,
                int(np.ceil(len(images) / batch_size)),
            ),
            flush=True,
        )

    summary: dict[str, dict] = {}
    for name, rec in accum.items():
        if not rec["cos"]:
            continue
        summary[name] = {
            "tensor": name,
            "op_type": "?",
            "shape": list(meta[name].shape) if name in meta else None,
            "cosine": float(np.mean(rec["cos"])),
            "cosine_min": float(np.min(rec["cos"])),
            "sqnr_db": float(np.mean(rec["sqnr"])),
            "rel_l2": float(np.mean(rec["rel"])),
            "absmax": float(np.mean(rec["absmax"])),
            "rms": float(np.mean(rec["rms"])),
        }
    return summary


def add_input_quant_error(
    variant_path: str, fp32_instr: str, images: list[np.ndarray]
) -> dict[str, float]:
    """Dequantized value of each QuantizeLinear input vs the FP32 tensor.

    Isolates *local* quantization damage from error inherited from earlier
    layers. If a layer's input quantization error is tiny but its output
    cosine is low, the problem is the weights, not the activations.
    """
    import onnx
    import onnxruntime as ort

    model = onnx.load(variant_path)
    targets = []
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and len(node.input) >= 2:
            targets.append((node.input[0], node.output[0]))
    if not targets:
        return {}

    # Expose each DequantizeLinear output so we can read the reconstructed value.
    deq_outputs = {n.output[0] for n in model.graph.node if n.op_type == "DequantizeLinear"}
    wanted = [(src, qout) for src, qout in targets]
    names = [qout for _src, qout in wanted]
    from onnx import helper

    existing = {o.name for o in model.graph.output}
    for name in names:
        if name in existing:
            continue
        try:
            model.graph.output.append(helper.make_empty_tensor_value_info(name))
        except Exception:  # noqa: BLE001
            pass
    tmp = variant_path + ".qi.onnx"
    onnx.save(model, tmp)

    fp32_sess = ort.InferenceSession(fp32_instr, providers=["CPUExecutionProvider"])
    input_name = fp32_sess.get_inputs()[0].name
    fp32_out_names = {o.name for o in fp32_sess.get_outputs()}

    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    sess_names = [o.name for o in sess.get_outputs()]
    usable = [
        (src, qout)
        for src, qout in wanted
        if qout in sess_names and src in fp32_out_names
    ]
    if not usable:
        os.remove(tmp)
        return {}

    batch = np.stack(images[: min(4, len(images))], axis=0)
    # Honour a fixed batch dimension in the graph.
    declared = fp32_sess.get_inputs()[0].shape[0]
    if isinstance(declared, int) and declared > 0 and len(batch) != declared:
        batch = np.stack(images[:declared], axis=0)
    fp32_vals = dict(
        zip(
            [o.name for o in fp32_sess.get_outputs()],
            fp32_sess.run([src for src, _ in usable], {input_name: batch}),
        )
    )
    qdq_vals = dict(zip(sess_names, sess.run([q for _s, q in usable], {input_name: batch})))

    result: dict[str, float] = {}
    for src, qout in usable:
        a, b = fp32_vals[src], qdq_vals[qout]
        if a.shape != b.shape:
            continue
        result[src] = {
            "input_cosine": cosine_similarity(a, b),
            "input_rel_l2": rel_l2(a, b),
        }
    try:
        os.remove(tmp)
    except OSError:
        pass
    return result


def find_first_break(rows: list[dict], threshold: float) -> dict | None:
    """First layer (in graph order) whose cosine similarity drops below threshold."""
    for row in rows:
        if row["cosine"] < threshold:
            return row
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        default=os.path.join(ROOT, "artifacts/models/mobilenet_v2_opset13_sim_preproc.onnx"),
    )
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "artifacts/models"))
    ap.add_argument("--calib", default=os.path.join(ROOT, "data/calib/calib_all.txt"))
    ap.add_argument("--num-images", type=int, default=32)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="requested batch; automatically clamped to the graph's fixed batch dim",
    )
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="variant labels; default is qdq_pertensor_minmax + qdq_perchannel_minmax",
    )
    ap.add_argument("--work-dir", default=os.path.join(ROOT, "artifacts/instrumented"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "artifacts"))
    args = ap.parse_args()

    if not os.path.exists(args.baseline):
        print("baseline not found:", args.baseline)
        return 1

    stem = "mobilenet_v2_opset13_sim"
    variants = args.variants or ["qdq_pertensor_minmax", "qdq_perchannel_minmax"]

    images = load_images(resolve_images(args.calib, args.num_images))
    print(f"images   : {len(images)}")
    if not images:
        print("no images loaded - check --calib")
        return 1

    print("instrumenting baseline ...", flush=True)
    variant_paths = [
        os.path.join(args.model_dir, f"{stem}_{label}.onnx")
        for label in variants
        if os.path.exists(os.path.join(args.model_dir, f"{stem}_{label}.onnx"))
    ]
    if not variant_paths:
        print("none of the requested variants exist in", args.model_dir)
        return 1

    # Only compare tensors that exist in *every* graph. ONNX Runtime's QDQ
    # quantizer swallows Clip nodes and reuses their output name on the Conv,
    # so comparing per-op outputs by name would silently pair a pre-activation
    # tensor with a post-activation one.
    shared = intersect_tensors(args.baseline, variant_paths)
    shared_names = [layer["tensor"] for layer in shared]
    print(f"  comparable tensors : {len(shared_names)} (intersection across "
          f"{len(variant_paths)} variants)")

    fp32_instr, fp32_meta = instrument_cached(
        args.baseline, args.work_dir, shared_names
    )
    print(f"  exposed {len(fp32_meta['layers'])} tensors")

    all_rows: list[dict] = []
    reports: dict[str, dict] = {}

    for label in variants:
        path = os.path.join(args.model_dir, f"{stem}_{label}.onnx")
        if not os.path.exists(path):
            print(f"skip {label}: {path} not found")
            continue
        print(f"\n=== {label} ===", flush=True)
        int8_instr, _ = instrument_cached(path, args.work_dir, shared_names)
        summary = compare_models(
            fp32_instr, int8_instr, images, batch_size=args.batch_size
        )

        order = {layer["tensor"]: layer["index"] for layer in fp32_meta["layers"]}
        op_types = {layer["tensor"]: layer["op_type"] for layer in fp32_meta["layers"]}
        for name, row in summary.items():
            row["op_type"] = op_types.get(name, "?")
            row["order"] = order.get(name, 10**6)
            row["variant"] = label

        rows = sorted(summary.values(), key=lambda r: r["order"])

        # fold in the isolated input-quantization error
        try:
            iqe = add_input_quant_error(path, fp32_instr, images)
            for row in rows:
                info = iqe.get(row["tensor"])
                row["input_cosine"] = info["input_cosine"] if info else None
                row["input_rel_l2"] = info["input_rel_l2"] if info else None
        except Exception as exc:  # noqa: BLE001
            print("  input-quant-error pass skipped:", type(exc).__name__, exc)
            for row in rows:
                row["input_cosine"] = None
                row["input_rel_l2"] = None

        first = find_first_break(rows, args.threshold)
        worst = min(rows, key=lambda r: r["cosine"])
        best = max(rows, key=lambda r: r["cosine"])
        print(f"  tensors compared : {len(rows)}")
        print(f"  mean cosine      : {np.mean([r['cosine'] for r in rows]):.6f}")
        print(f"  worst tensor     : {worst['cosine']:.6f}  {worst['tensor']}")
        print(f"  BEST tensor      : {best['cosine']:.6f}  {best['tensor']}")
        if first:
            print(
                "  FIRST below %.2f  : cos=%.6f  %s (%s)  [layer #%d]"
                % (args.threshold, first["cosine"], first["tensor"],
                   first["op_type"], first["order"])
            )
        else:
            print(f"  no tensor below {args.threshold} - quantization is well behaved")

        reports[label] = {
            "num_tensors": len(rows),
            "mean_cosine": float(np.mean([r["cosine"] for r in rows])),
            "worst": worst,
            "best": best,
            "first_break": first,
            "threshold": args.threshold,
        }
        all_rows.extend(rows)

    if not all_rows:
        print("\nno results produced")
        return 1

    columns = [
        "variant", "order", "op_type", "tensor", "shape",
        "cosine", "cosine_min", "sqnr_db", "rel_l2", "absmax",
        "input_cosine", "input_rel_l2",
    ]
    csv_path = os.path.join(args.outdir, "tables", "04_layer_metrics.csv")
    write_csv(all_rows, csv_path)
    write_json(reports, os.path.join(args.outdir, "reports", "04_layer_analysis.json"))

    print("\n=== 逐层对比：最差的 20 层 ===")
    worst_rows = sorted(all_rows, key=lambda r: r["cosine"])[:20]
    print(
        format_table(
            [
                {
                    "variant": r["variant"],
                    "order": r["order"],
                    "op": r["op_type"],
                    "tensor": (r["tensor"] or "")[:38],
                    "cosine": r["cosine"],
                    "sqnr_db": r["sqnr_db"],
                    "input_cos": r["input_cosine"],
                }
                for r in worst_rows
            ],
            ["variant", "order", "op", "tensor", "cosine", "sqnr_db", "input_cos"],
        )
    )

    # per-tensor vs per-channel head-to-head
    if "qdq_pertensor_minmax" in reports and "qdq_perchannel_minmax" in reports:
        load_q = {r["tensor"]: r for r in all_rows if r["variant"] == "qdq_pertensor_minmax"}
        pc_q = {r["tensor"]: r for r in all_rows if r["variant"] == "qdq_perchannel_minmax"}
        common = sorted(set(load_q) & set(pc_q), key=lambda n: load_q[n]["order"])
        better = [n for n in common if pc_q[n]["cosine"] > load_q[n]["cosine"]]
        print("\n=== per-tensor vs per-channel ===")
        print(f"  共同层数            : {len(common)}")
        print(f"  per-channel 更好的层 : {len(better)} / {len(common)}")
        print(f"  平均 cosine (per-tensor) : {np.mean([load_q[n]['cosine'] for n in common]):.6f}")
        print(f"  平均 cosine (per-channel): {np.mean([pc_q[n]['cosine'] for n in common]):.6f}")
        print("\n  差距最大的 10 层 (per-channel 优势):")
        delta = sorted(
            common, key=lambda n: pc_q[n]["cosine"] - load_q[n]["cosine"]
        )[:10]
        print(
            format_table(
                [
                    {
                        "order": load_q[n]["order"],
                        "op": load_q[n]["op_type"],
                        "tensor": (n or "")[:38],
                        "cos_pertensor": load_q[n]["cosine"],
                        "cos_perchannel": pc_q[n]["cosine"],
                        "delta": pc_q[n]["cosine"] - load_q[n]["cosine"],
                    }
                    for n in delta
                ],
                ["order", "op", "tensor", "cos_pertensor", "cos_perchannel", "delta"],
            )
        )

    print(f"\ncsv      : {os.path.relpath(csv_path, ROOT)}")
    print(f"report   : {os.path.relpath(os.path.join(args.outdir, 'reports', '04_layer_analysis.json'), ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
