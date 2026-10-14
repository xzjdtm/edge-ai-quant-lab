"""W1 quantization-error study: ONNX export, PTQ quantization, layer analysis.

Scripts are numbered in execution order:

* ``01_export_onnx.py``      PyTorch -> ONNX (+ onnxsim), records graph stats
* ``02_prepare_calib.py``    build real calibration image sets
* ``02a_fetch_images.py``    download real images (mirror speed testing)
* ``02b_fetch_github_images.py``  fetch ImageNet sample images via GitHub raw
* ``03_quantize.py``         FP32 -> INT8 PTQ (QDQ + QOperator, per-tensor/channel)
* ``04_layer_analysis.py``   per-layer cosine similarity / SQNR, find the layer that breaks
* ``05_accuracy.py``         classification accuracy and confidence drift
* ``06_calib_sweep.py``      accuracy vs calibration-set size and diversity
* ``07_make_report.py``      collect everything into figures + report
* ``08_ort_inference.py``    ONNX Runtime session/provider/IO-binding/benchmark

``common.py`` holds the metrics, graph instrumentation and reporting helpers.
"""