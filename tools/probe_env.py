import importlib.util as u
import sys

mods = [
    "torch", "torchvision", "onnx", "onnxruntime", "onnxsim",
    "cv2", "numpy", "matplotlib", "sklearn", "pandas",
    "rknn", "rknn_toolkit2", "ultralytics", "netron",
]

print("python", sys.version.split()[0])
for m in mods:
    spec = u.find_spec(m)
    if spec is None:
        print("  %-16s MISSING" % m)
        continue
    ver = ""
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "")
        if not ver:
            ver = getattr(mod, "version", "")
    except Exception as ex:
        ver = "(import error: %s)" % type(ex).__name__
    print("  %-16s OK   %s" % (m, ver))
