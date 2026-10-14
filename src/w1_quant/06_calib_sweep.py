"""W1-D6 experiment 2: accuracy vs calibration-set size and diversity.

The interview question is "校准集怎么选". Hand-waving is not an answer; this
produces the curve.

Two independent variables, deliberately separated:

* **size**      - 10 / 50 / 200 / 500 images, all balanced across classes
* **diversity** - same 500 images, but drawn from one class only, or half-half

The second axis is the interesting one. A calibration set that is
*representative* matters more than one that is merely *large*: activation
ranges observed from a single class are narrower than the real deployment
distribution, so the chosen scale clips activations that appear at inference
time.

Metrics per point: mean per-layer cosine similarity, SQNR, top-1 agreement with
FP32, and confidence drop.

Usage
-----
    python 06_calib_sweep.py --num-eval-images 100
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
    effective_batch,
    format_table,
    instrument_explicit,
    intersect_tensors,
    load_layer_meta,
    redirect_tempdir,
    sqnr,
    write_csv,
    write_json,
)

TMPDIR = redirect_tempdir()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG")


# --------------------------------------------------------------------------
# image loading
# --------------------------------------------------------------------------


def resolve_images(spec: str, limit: int | None = None) -> list[str]:
    if os.path.isdir(spec):
        paths: list[str] = []
        for ext in IMAGE_EXTS:
            paths.extend(glob.glob(os.path.join(spec, "**", ext), recursive=True))
        paths = sorted(set(paths))
    else:
        with open(spec, encoding="utf-8") as fh:
            paths = [line.strip() for line in fh if line.strip()]
    return paths[:limit] if limit else paths


def load_images(paths: list[str], size: int = 224) -> list[np.ndarray]:
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
        out.append(np.transpose((img.astype(np.float32) / 255.0 - mean) / std, (2, 0, 1)))
    return out


class Reader:
    """Minimal onnxruntime calibration reader over in-memory arrays."""

    def __init__(self, arrays: list[np.ndarray], input_name: str):
        self.arrays = arrays
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.arrays):
            return None
        value = self.arrays[self.i : self.i + 1]
        self.i += 1
        return {self.input_name: np.stack(value, axis=0)}

    def rewind(self):
        self.i = 0


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def instrument_cached(model_path: str, work_dir: str, tensors: list[str]):
    os.makedirs(work_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    out = os.path.join(work_dir, f"{stem}_shared.onnx")
    if not os.path.exists(out):
        meta = instrument_explicit(model_path, out, tensors)
    else:
        meta = load_layer_meta(out)
    return out, meta


def activation_metrics(
    fp32_instr: str, int8_instr: str, images: list[np.ndarray], chunk: int = 24
) -> dict:
    import onnxruntime as ort

    s_f = ort.InferenceSession(fp32_instr, providers=["CPUExecutionProvider"])
    s_q = ort.InferenceSession(int8_instr, providers=["CPUExecutionProvider"])
    names = [o.name for o in s_f.get_outputs()]
    shared = [n for n in names if n in {o.name for o in s_q.get_outputs()}]
    input_name = s_f.get_inputs()[0].name
    batch_size = effective_batch(s_f, 1)

    cosines, sqnrs = [], []
    for start in range(0, len(images), batch_size):
        batch = np.stack(images[start : start + batch_size], axis=0)
        for offset in range(0, len(shared), chunk):
            group = shared[offset : offset + chunk]
            a_list = s_f.run(group, {input_name: batch})
            b_list = s_q.run(group, {input_name: batch})
            for a, b in zip(a_list, b_list):
                if a.shape != b.shape:
                    continue
                cosines.append(cosine_similarity(a, b))
                sqnrs.append(sqnr(a, b))
    return {
        "mean_cosine": float(np.mean(cosines)) if cosines else float("nan"),
        "min_cosine": float(np.min(cosines)) if cosines else float("nan"),
        "mean_sqnr_db": float(np.mean(sqnrs)) if sqnrs else float("nan"),
        "num_tensors": len(cosines),
    }


def task_metrics(fp32_path: str, int8_path: str, images: list[np.ndarray], batch: int = 8) -> dict:
    import onnxruntime as ort

    def run(path):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        step = effective_batch(sess, batch)
        return np.concatenate(
            [
                np.asarray(sess.run(None, {name: np.stack(images[s : s + step], axis=0)})[0],
                           dtype=np.float64)
                for s in range(0, len(images), step)
            ],
            axis=0,
        )

    def softmax(x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    ref, got = run(fp32_path), run(int8_path)
    p_ref, p_got = softmax(ref), softmax(got)
    top_ref, top_got = np.argmax(p_ref, 1), np.argmax(p_got, 1)
    eps = 1e-12
    kl = float(
        np.mean(np.sum(p_ref * (np.log(p_ref + eps) - np.log(p_got + eps)), axis=1))
    )
    return {
        "top1_match": float(np.mean(top_ref == top_got)),
        "mean_kl": kl,
        "conf_fp32": float(np.mean(p_ref[np.arange(len(top_ref)), top_ref])),
        "conf_int8": float(np.mean(p_got[np.arange(len(top_got)), top_got])),
    }


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        default=os.path.join(ROOT, "artifacts/models/mobilenet_v2_opset13_sim_preproc.onnx"),
    )
    ap.add_argument("--calib-dir", default=os.path.join(ROOT, "data/calib"))
    ap.add_argument("--num-eval-images", type=int, default=64)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--work-dir", default=os.path.join(ROOT, "artifacts/instrumented"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.baseline):
        print("baseline not found:", args.baseline)
        return 1

    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    calib_dir = args.calib_dir
    points: list[tuple[str, str, str]] = []  # (label, list file, group)

    for size in (10, 50, 200, 500):
        path = os.path.join(calib_dir, f"calib_nested_{size}.txt")
        if os.path.exists(path):
            points.append((f"size_{size}", path, "size"))
    for name, label in (
        ("calib_narrow_500.txt", "narrow_500"),
        ("calib_mixed_500.txt", "mixed_500"),
        ("calib_domain_balanced_500.txt", "domain_balanced_500"),
    ):
        path = os.path.join(calib_dir, name)
        if os.path.exists(path):
            points.append((label, path, "diversity"))

    if not points:
        print("no calibration lists found in", calib_dir)
        print("run 02_prepare_calib.py first")
        return 1

    # Evaluation images: a held-out slice of all images, independent of the
    # calibration lists so size sweeps are not scored on their own data.
    all_list = os.path.join(calib_dir, "all_images.txt")
    eval_paths = (
        resolve_images(all_list)[-args.num_eval_images :]
        if os.path.exists(all_list)
        else resolve_images(points[0][1])
    )
    eval_images = load_images(eval_paths)
    print(f"eval images: {len(eval_images)}")

    import onnxruntime as ort

    sess = ort.InferenceSession(args.baseline, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    model_dir = os.path.join(args.outdir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # ---- stage 1: quantize every calibration variant ----------------------
    quantized: list[tuple[str, str, str, int, str]] = []  # label, group, path, n, list_file
    for label, list_file, group in points:
        out_model = os.path.join(model_dir, f"mobilenet_v2_calib_{label}.onnx")
        calib_paths = resolve_images(list_file)
        n_calib = len(calib_paths)
        print(f"\n=== {label} ({group}) : {n_calib} calibration images ===", flush=True)

        if args.force or not os.path.exists(out_model):
            arrays = load_images(calib_paths)
            if not arrays:
                print("  no calibration images loaded, skipping")
                continue
            reader = Reader(arrays, input_name)
            quantize_static(
                args.baseline,
                out_model,
                reader,
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                calibrate_method=CalibrationMethod.MinMax,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QInt8,
            )
            print("  quantized -> %s" % os.path.basename(out_model))
        else:
            print("  exists, reusing")
        quantized.append((label, group, out_model, n_calib, list_file))

    if not quantized:
        print("no models available")
        return 1

    # ---- stage 2: one intersection, then instrument everything ------------
    # Tensors must exist in the FP32 baseline *and* every quantized variant, so
    # the comparison is on the same computation everywhere.
    shared = intersect_tensors(args.baseline, [q[2] for q in quantized])
    shared_names = [layer["tensor"] for layer in shared]
    print(f"\ncomparable tensors : {len(shared_names)}")
    fp32_instr, _ = instrument_cached(args.baseline, args.work_dir, shared_names)

    # ---- stage 3: measure -------------------------------------------------
    rows = []
    for label, group, out_model, n_calib, _list_file in quantized:
        print(f"\n--- measuring {label} ---", flush=True)
        int8_instr, _ = instrument_cached(out_model, args.work_dir, shared_names)
        act = activation_metrics(fp32_instr, int8_instr, eval_images)
        task = task_metrics(args.baseline, out_model, eval_images)

        row = {
            "label": label,
            "group": group,
            "calib_images": n_calib,
            "file_mb": round(os.path.getsize(out_model) / 1e6, 2),
            **act,
            **task,
            "conf_drop": task["conf_fp32"] - task["conf_int8"],
        }
        rows.append(row)
        print(
            "  mean_cosine=%.5f  min_cosine=%.5f  SQNR=%.2f dB  top1_match=%.4f  KL=%.5f"
            % (row["mean_cosine"], row["min_cosine"], row["mean_sqnr_db"],
               row["top1_match"], row["mean_kl"])
        )

    if not rows:
        print("no results")
        return 1

    write_csv(rows, os.path.join(args.outdir, "tables", "06_calib_sweep.csv"))
    write_json(
        {"eval_images": len(eval_images), "points": rows},
        os.path.join(args.outdir, "reports", "06_calib_sweep.json"),
    )

    size_rows = sorted([r for r in rows if r["group"] == "size"], key=lambda r: r["calib_images"])
    div_rows = [r for r in rows if r["group"] == "diversity"]

    if size_rows:
        print("\n" + format_table(
            [
                {
                    "calib_images": r["calib_images"],
                    "mean_cosine": r["mean_cosine"],
                    "min_cosine": r["min_cosine"],
                    "SQNR_dB": r["mean_sqnr_db"],
                    "top1_match": r["top1_match"],
                    "KL": r["mean_kl"],
                    "conf_drop": r["conf_drop"],
                }
                for r in size_rows
            ],
            ["calib_images", "mean_cosine", "min_cosine", "SQNR_dB", "top1_match", "KL", "conf_drop"],
            title="=== 实验A：校准集规模 vs 精度 ===",
        ))
        if len(size_rows) >= 2:
            first, last = size_rows[0], size_rows[-1]
            gain = last["mean_cosine"] - first["mean_cosine"]
            print(
                "  从 %d 张到 %d 张：mean_cosine 提升 %.5f, SQNR 提升 %.2f dB"
                % (first["calib_images"], last["calib_images"],
                   gain, last["mean_sqnr_db"] - first["mean_sqnr_db"])
            )
            saturated = None
            for i in range(1, len(size_rows)):
                if size_rows[i]["mean_cosine"] - size_rows[i - 1]["mean_cosine"] < 0.001:
                    saturated = size_rows[i]["calib_images"]
                    break
            print("  收益饱和点：%s" % (f"{saturated} 张左右" if saturated else "尚未饱和"))

    if div_rows:
        print("\n" + format_table(
            [
                {
                    "label": r["label"],
                    "calib_images": r["calib_images"],
                    "mean_cosine": r["mean_cosine"],
                    "min_cosine": r["min_cosine"],
                    "SQNR_dB": r["mean_sqnr_db"],
                    "top1_match": r["top1_match"],
                    "KL": r["mean_kl"],
                }
                for r in div_rows
            ],
            ["label", "calib_images", "mean_cosine", "min_cosine", "SQNR_dB", "top1_match", "KL"],
            title="=== 实验B：校准集多样性（同样 500 张）===",
        ))
        balanced = next((r for r in div_rows if r["label"] == "size_500"), None)
        if balanced:
            for row in div_rows:
                if row["label"] == "size_500":
                    continue
                print(
                    "  %s 相对均衡校准集：mean_cosine %+.5f, top1_match %+.4f, KL %+.5f"
                    % (row["label"],
                       row["mean_cosine"] - balanced["mean_cosine"],
                       row["top1_match"] - balanced["top1_match"],
                       row["mean_kl"] - balanced["mean_kl"])
                )

    print(f"\ncsv      : {os.path.relpath(os.path.join(args.outdir, 'tables', '06_calib_sweep.csv'), ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
