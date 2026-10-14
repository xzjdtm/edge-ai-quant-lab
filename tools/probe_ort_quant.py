"""Probe ONNX Runtime quantization API surface.

Determines which quantization entry points and options are available in the
installed onnxruntime version, so the experiment scripts target real APIs.
"""

import inspect


def show(label, obj):
    print("=" * 70)
    print(label)
    print("=" * 70)
    try:
        print("signature:", inspect.signature(obj))
    except (TypeError, ValueError) as exc:
        print("signature unavailable:", exc)


print("onnxruntime version:", end=" ")
try:
    import onnxruntime as ort
    print(ort.__version__)
except Exception as exc:
    print("IMPORT FAILED:", exc)
    raise SystemExit(1)

try:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    print("\nTop-level quantization imports: OK")
    print("QuantFormat members :", [m.name for m in QuantFormat])
    print("QuantType members   :", [m.name for m in QuantType])
    print("CalibrationMethod   :", [m.name for m in CalibrationMethod])

    show("quantize_static", quantize_static)
    show("quantize_dynamic", quantize_dynamic)
except Exception as exc:
    print("\nTop-level quantization imports FAILED:", type(exc).__name__, exc)

print()
try:
    from onnxruntime.quantization import QConfig

    show("QConfig.__init__", QConfig.__init__)
    print("QConfig attributes:", [a for a in dir(QConfig) if not a.startswith("_")])
except Exception as exc:
    print("QConfig unavailable:", type(exc).__name__, exc)

print()
try:
    from onnxruntime.quantization import qdq_quantize

    show("qdq_quantize", qdq_quantize)
except Exception as exc:
    print("qdq_quantize unavailable:", type(exc).__name__, exc)

print()
try:
    from onnxruntime.quantization import CalibrationDataReader

    show("CalibrationDataReader", CalibrationDataReader)
except Exception as exc:
    print("CalibrationDataReader unavailable:", type(exc).__name__, exc)

print()
try:
    from onnxruntime.quantization import quant_utils

    print("quant_utils members:", [a for a in dir(quant_utils) if not a.startswith("_")][:40])
except Exception as exc:
    print("quant_utils unavailable:", type(exc).__name__, exc)

print()
try:
    import onnx

    print("onnx version:", onnx.__version__)
except Exception as exc:
    print("onnx unavailable:", type(exc).__name__, exc)

print()
try:
    import onnxsim

    print("onnxsim version:", getattr(onnxsim, "__version__", "unknown"))
except Exception as exc:
    print("onnxsim unavailable:", type(exc).__name__, exc)
