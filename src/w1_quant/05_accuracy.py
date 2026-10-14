"""W1-D5/D6: what does INT8 quantization actually cost in accuracy?

Layer-wise cosine similarity measures *activation drift*. This script measures
what that drift costs at the task level:

* ``top1_match``      - how often INT8 picks the same class as FP32
* ``top1_confidence_drop`` - how much probability mass the winner loses
* ``kl_divergence``   - full-distribution drift, not just the argmax
* ``mean_abs_logit_diff`` - raw logit movement

The distinction matters in an interview: a model can keep the same argmax while
its confidence collapses, which then breaks downstream thresholds, NMS or
"reject if uncertain" logic in a real edge pipeline.

Usage
-----
    python 05_accuracy.py --num-images 200
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w1_quant.common import (  # noqa: E402
    effective_batch,
    format_table,
    redirect_tempdir,
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


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def load_batch(paths: list[str], size: int = 224) -> tuple[np.ndarray, list[str]]:
    import cv2

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arrays, kept = [], []
    for path in paths:
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            continue
        if img is None:
            continue
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size))
        arrays.append(np.transpose((img.astype(np.float32) / 255.0 - mean) / std, (2, 0, 1)))
        kept.append(path)
    return np.stack(arrays, axis=0), kept


def evaluate(
    variant_path: str, images: np.ndarray, reference: np.ndarray, batch: int
) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(variant_path, opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    batch = effective_batch(sess, batch)

    logits = []
    for start in range(0, len(images), batch):
        out = sess.run(None, {input_name: images[start : start + batch]})[0]
        logits.append(np.asarray(out, dtype=np.float64))
    logits = np.concatenate(logits, axis=0)

    probs_q = softmax(logits)
    probs_f = softmax(reference)

    top1_f = np.argmax(probs_f, axis=1)
    top1_q = np.argmax(probs_q, axis=1)

    conf_f = probs_f[np.arange(len(top1_f)), top1_f]
    conf_q = probs_q[np.arange(len(top1_q)), top1_q]

    eps = 1e-12
    kl = np.sum(
        probs_f * (np.log(probs_f + eps) - np.log(probs_q + eps)), axis=1
    )

    # Is the FP32 winner still inside the INT8 top-5?
    order_q = np.argsort(-probs_q, axis=1)[:, :5]
    top5_hit = np.array([top1_f[i] in order_q[i] for i in range(len(top1_f))])

    return {
        "top1_match": float(np.mean(top1_q == top1_f)),
        "top5_contains_fp32_top1": float(np.mean(top5_hit)),
        "mean_confidence_fp32": float(np.mean(conf_f)),
        "mean_confidence_int8": float(np.mean(conf_q)),
        "confidence_drop": float(np.mean(conf_f) - np.mean(conf_q)),
        "mean_kl_divergence": float(np.mean(kl)),
        "max_kl_divergence": float(np.max(kl)),
        "mean_abs_logit_diff": float(np.mean(np.abs(logits - reference))),
        "max_abs_logit_diff": float(np.max(np.abs(logits - reference))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        default=os.path.join(ROOT, "artifacts/models/mobilenet_v2_opset13_sim_preproc.onnx"),
    )
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "artifacts/models"))
    ap.add_argument("--calib", default=os.path.join(ROOT, "data/calib/all_images.txt"))
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "artifacts"))
    args = ap.parse_args()

    if not os.path.exists(args.baseline):
        print("baseline not found:", args.baseline)
        return 1

    variants = args.variants or [
        "qdq_pertensor_minmax",
        "qdq_perchannel_minmax",
        "qdq_perchannel_entropy",
        "qdq_perchannel_percentile",
        "qoperator_pertensor",
        "qoperator_perchannel",
        "qdq_perchannel_narrowcalib",
    ]

    paths = resolve_images(args.calib, args.num_images)
    print(f"images   : {len(paths)}")
    images, kept = load_batch(paths)
    print(f"loaded   : {images.shape}")

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    ref_sess = ort.InferenceSession(args.baseline, opts, providers=["CPUExecutionProvider"])
    input_name = ref_sess.get_inputs()[0].name
    args.batch_size = effective_batch(ref_sess, args.batch_size)
    print(f"batch    : {args.batch_size} (clamped to the graph's fixed batch dim)")
    ref_logits = np.concatenate(
        [
            np.asarray(ref_sess.run(None, {input_name: images[s : s + args.batch_size]})[0],
                       dtype=np.float64)
            for s in range(0, len(images), args.batch_size)
        ],
        axis=0,
    )
    print(f"baseline logits: {ref_logits.shape}")

    rows = []
    print("\n=== 精度评估 ===")
    for label in variants:
        path = os.path.join(args.model_dir, f"mobilenet_v2_opset13_sim_{label}.onnx")
        if not os.path.exists(path):
            print(f"  skip {label} (missing)")
            continue
        try:
            metrics = evaluate(path, images, ref_logits, args.batch_size)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {label}: {type(exc).__name__}: {exc}")
            continue
        metrics["label"] = label
        metrics["file_mb"] = round(os.path.getsize(path) / 1e6, 2)
        rows.append(metrics)
        print(
            "  %-28s top1_match=%.4f  conf_drop=%+.4f  KL=%.5f  maxlogit=%.4f"
            % (label, metrics["top1_match"], metrics["confidence_drop"],
               metrics["mean_kl_divergence"], metrics["max_abs_logit_diff"])
        )

    if not rows:
        print("\nno variants evaluated")
        return 1

    outdir = args.outdir
    write_csv(rows, os.path.join(outdir, "tables", "05_accuracy.csv"))
    write_json(
        {"num_images": len(images), "variants": rows},
        os.path.join(outdir, "reports", "05_accuracy.json"),
    )

    print("\n" + format_table(
        [
            {
                "variant": r["label"],
                "size_MB": r["file_mb"],
                "top1_match": r["top1_match"],
                "conf_fp32": r["mean_confidence_fp32"],
                "conf_int8": r["mean_confidence_int8"],
                "KL": r["mean_kl_divergence"],
                "max_logit_diff": r["max_abs_logit_diff"],
            }
            for r in rows
        ],
        ["variant", "size_MB", "top1_match", "conf_fp32", "conf_int8", "KL", "max_logit_diff"],
        title="=== 量化变体精度对比 ===",
    ))
    print(f"\ncsv      : {os.path.relpath(os.path.join(outdir, 'tables', '05_accuracy.csv'), ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
