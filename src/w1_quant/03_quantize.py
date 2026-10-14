"""W1-D5/D6: FP32 -> INT8 post-training quantization (PTQ).

Produces a family of quantized models so the D6 experiments have real
comparisons instead of a single "it works" run:

* ``baseline_preproc`` - FP32 graph after ONNX Runtime's ``quant_pre_process``.
  This is the *fair* reference: quantization also runs pre-processing, so
  comparing INT8 against the raw export would mostly measure graph rewriting.
* ``qdq_pertensor``    - QDQ format, one scale per tensor.
* ``qdq_perchannel``   - QDQ format, one scale per output channel.
* ``qoperator_*``      - QOperator format, kept because real NPU toolchains
  (RKNN, nncase) consume similar fused forms and the comparison is instructive.
* calibration-method variants (MinMax / Entropy / Percentile) and an
  unrepresentative single-class calibration set.

Notes that matter in an interview
---------------------------------
* ``per_channel`` only applies to **weights**; activations stay per-tensor
  because their range is data dependent.
* ``reduce_range`` shrinks the INT8 range to avoid saturation on x86 VNNI-style
  paths; it costs range, so we keep it off.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w1_quant.common import redirect_tempdir, write_csv, write_json  # noqa: E402

# Must happen before onnxruntime/onnx create any temp file.
TMPDIR = redirect_tempdir()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG")


# --------------------------------------------------------------------------
# calibration data reader
# --------------------------------------------------------------------------


class ImageCalibrationReader:
    """Yields ``{input_name: NCHW float32}`` batches from a list of images.

    onnxruntime only requires an object with ``get_next()``; it does not have to
    subclass ``CalibrationDataReader``, though this mirrors its contract.
    """

    def __init__(
        self,
        image_paths: list[str],
        input_name: str,
        input_size: int = 224,
        batch_size: int = 1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ) -> None:
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.input_size = input_size
        self.batch_size = max(1, batch_size)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.cursor = 0
        self.failed: list[str] = []

    def _load(self, path: str) -> np.ndarray | None:
        import cv2

        # cv2.imread cannot handle non-ASCII paths on Windows.
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            return None
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        x = img.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return np.transpose(x, (2, 0, 1))  # HWC -> CHW

    def get_next(self):
        if self.cursor >= len(self.image_paths):
            return None
        batch = self.image_paths[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        arrays = []
        for path in batch:
            arr = self._load(path)
            if arr is None:
                self.failed.append(path)
                continue
            arrays.append(arr)
        if not arrays:
            return None
        return {self.input_name: np.stack(arrays, axis=0)}

    @property
    def loaded(self) -> int:
        return self.cursor - len(self.failed)


def read_image_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def resolve_input(image_dir: str) -> tuple[list[str], bool]:
    """Return (image paths, is_directory_mode)."""
    if os.path.isdir(image_dir):
        paths: list[str] = []
        for ext in IMAGE_EXTS:
            paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
        return sorted(set(paths)), True
    if image_dir.endswith(".txt"):
        return read_image_list(image_dir), False
    if os.path.isfile(image_dir):
        return [image_dir], False
    raise FileNotFoundError(image_dir)


# --------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------


def quantization_node_count(path: str) -> dict:
    import onnx

    model = onnx.load(path)
    q = sum(1 for n in model.graph.node if n.op_type == "QuantizeLinear")
    dq = sum(1 for n in model.graph.node if n.op_type == "DequantizeLinear")
    qlinear = sum(
        1 for n in model.graph.node if n.op_type in ("QLinearConv", "QLinearMatMul")
    )
    return {
        "file_mb": round(os.path.getsize(path) / 1e6, 2),
        "nodes": len(model.graph.node),
        "quantize_linear": q,
        "dequantize_linear": dq,
        "qlinear_fused": qlinear,
        "qdq_style": bool(q or dq),
    }


def run_quantize(
    src: str,
    dst: str,
    reader,
    *,
    quant_format,
    per_channel: bool,
    calibrate_method,
    weight_type,
    activation_type,
    reduce_range: bool = False,
    extra_options: dict | None = None,
) -> dict:
    from onnxruntime.quantization import quantize_static

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    quantize_static(
        src,
        dst,
        reader,
        quant_format=quant_format,
        per_channel=per_channel,
        calibrate_method=calibrate_method,
        weight_type=weight_type,
        activation_type=activation_type,
        reduce_range=reduce_range,
        extra_options=extra_options or {},
    )
    info = quantization_node_count(dst)
    info["path"] = dst
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default=os.path.join(ROOT, "artifacts/models/mobilenet_v2_opset13_sim.onnx")
    )
    ap.add_argument(
        "--calib", default=os.path.join(ROOT, "data/calib/calib_all.txt")
    )
    ap.add_argument(
        "--narrow-calib", default=os.path.join(ROOT, "data/calib/calib_narrow_500.txt")
    )
    ap.add_argument("--outdir", default=os.path.join(ROOT, "artifacts/models"))
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=500, help="max calibration images")
    ap.add_argument("--force", action="store_true", help="re-quantize existing files")
    ap.add_argument(
        "--report", default=os.path.join(ROOT, "artifacts/reports/03_quant_report.json")
    )
    args = ap.parse_args()

    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quant_pre_process,
    )

    if not os.path.exists(args.model):
        print("model not found:", args.model)
        print("run 01_export_onnx.py first")
        return 1
    if not os.path.exists(args.calib):
        print("calibration list not found:", args.calib)
        print("run 02_prepare_calib.py first")
        return 1

    import onnxruntime as ort

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.model))[0]

    # ---- baseline: FP32, but pre-processed like the quantizer would do -----
    baseline = os.path.join(outdir, f"{stem}_preproc.onnx")
    preproc_info = {}
    if args.force or not os.path.exists(baseline):
        print("quant_pre_process ->", os.path.basename(baseline))
        try:
            # symbolic_shape=False keeps fixed shapes, matching the RKNN flow
            quant_pre_process(args.model, baseline, skip_symbolic_shape=True)
        except TypeError:
            quant_pre_process(args.model, baseline)
        import onnx

        m = onnx.load(baseline)
        preproc_info = {"nodes": len(m.graph.node), "file_mb": round(os.path.getsize(baseline) / 1e6, 2)}
    else:
        import onnx

        m = onnx.load(baseline)
        preproc_info = {"nodes": len(m.graph.node), "file_mb": round(os.path.getsize(baseline) / 1e6, 2)}
        print("baseline exists:", os.path.basename(baseline))

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"input    : {input_name}")

    calib_paths, dir_mode = resolve_input(args.calib)
    calib_paths = calib_paths[: args.limit]
    print(f"calib    : {len(calib_paths)} images (dir_mode={dir_mode})")

    narrow_paths: list[str] = []
    if args.narrow_calib and os.path.exists(args.narrow_calib):
        narrow_paths = resolve_input(args.narrow_calib)[0][: args.limit]
        print(f"narrow   : {len(narrow_paths)} images")

    def make_reader(paths):
        return ImageCalibrationReader(
            paths, input_name, input_size=args.input_size, batch_size=1
        )

    # ------------------------------------------------------------------ plan
    plan = [
        # label,                      format,                   per_ch, method,               source
        ("qdq_pertensor_minmax", QuantFormat.QDQ, False, CalibrationMethod.MinMax, "all"),
        ("qdq_perchannel_minmax", QuantFormat.QDQ, True, CalibrationMethod.MinMax, "all"),
        ("qdq_perchannel_entropy", QuantFormat.QDQ, True, CalibrationMethod.Entropy, "all"),
        ("qdq_perchannel_percentile", QuantFormat.QDQ, True, CalibrationMethod.Percentile, "all"),
        ("qoperator_pertensor", QuantFormat.QOperator, False, CalibrationMethod.MinMax, "all"),
        ("qoperator_perchannel", QuantFormat.QOperator, True, CalibrationMethod.MinMax, "all"),
    ]
    if narrow_paths:
        plan.append(
            ("qdq_perchannel_narrowcalib", QuantFormat.QDQ, True, CalibrationMethod.MinMax, "narrow")
        )

    results = []
    for label, fmt, per_channel, method, source in plan:
        dst = os.path.join(outdir, f"{stem}_{label}.onnx")
        if os.path.exists(dst) and not args.force:
            info = quantization_node_count(dst)
            info.update(path=dst)
            print(f"skip     : {label} (exists)")
        else:
            paths = narrow_paths if source == "narrow" else calib_paths
            print(f"quantize : {label} ({fmt.name}, per_channel={per_channel}, "
                  f"{method.name}, {source}) ...", flush=True)
            reader = make_reader(paths)
            try:
                info = run_quantize(
                    baseline,
                    dst,
                    reader,
                    quant_format=fmt,
                    per_channel=per_channel,
                    calibrate_method=method,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                if reader.loaded == 0:
                    print(f"  cause: none of the {len(paths)} calibration images could be "
                          f"decoded (first path: {paths[0] if paths else 'n/a'})")
                continue
            if reader.loaded == 0:
                print("  WARNING: quantized without reading any calibration image")
            elif reader.failed:
                print(f"  note: {len(reader.failed)} of {len(paths)} calibration images "
                      f"could not be decoded")
            print("  -> %.2f MB, Q=%d DQ=%d QLinear=%d"
                  % (info["file_mb"], info["quantize_linear"],
                     info["dequantize_linear"], info["qlinear_fused"]))
        info.update(
            label=label,
            quant_format=fmt.name,
            per_channel=per_channel,
            calibrate_method=method.name,
            calib_source=source,
            calib_images=len(narrow_paths if source == "narrow" else calib_paths),
        )
        results.append(info)

    report = {
        "source_model": args.model,
        "baseline_preproc": dict(preproc_info, path=baseline),
        "calibration": {
            "all": {"path": args.calib, "images": len(calib_paths)},
            "narrow": {"path": args.narrow_calib, "images": len(narrow_paths)},
        },
        "variants": results,
    }
    write_json(report, args.report)
    write_csv(results, os.path.join(ROOT, "artifacts/tables/03_quant_variants.csv"))

    print("\n=== 量化变体一览 ===")
    print("%-28s %-10s %-9s %-11s %8s %6s %7s"
          % ("label", "format", "per_ch", "calib_method", "size_MB", "Q", "QLinear"))
    for row in results:
        print("%-28s %-10s %-9s %-11s %8.2f %6d %7d"
              % (row["label"], row["quant_format"], row["per_channel"],
                 row["calibrate_method"], row["file_mb"],
                 row["quantize_linear"], row["qlinear_fused"]))
    print(f"\nreport   : {os.path.relpath(args.report, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
