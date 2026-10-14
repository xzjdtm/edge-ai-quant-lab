"""Shared helpers for the W1 quantization-error study.

Design notes
------------
Two models are compared for every experiment:

* **FP32**  - the ONNX model after export (+ onnxsim) and, for a fair graph
  shape, after ONNX Runtime's `quant_pre_process`.
* **INT8**  - the same model quantized with `quantize_static` in **QDQ** format.

QDQ is used deliberately. Unlike the QOperator format it keeps the original
Conv/MatMul nodes and merely wraps their inputs with QuantizeLinear /
DequantizeLinear pairs. That means tensor names stay comparable between the two
graphs, which is what makes layer-by-layer analysis possible at all.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# --------------------------------------------------------------------------
# environment workarounds
# --------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def redirect_tempdir() -> str:
    """Point `tempfile` at a workspace directory whose dirs are writable.

    Two sandbox behaviours have to be worked around, both verified by direct
    experiment on this machine:

    1. ``TEMP`` is injected into every child process and cannot be overridden via
       the environment. Files placed there cannot be read back, which breaks
       ``quant_pre_process`` (it stages ``optimized.onnx`` in a temp dir and then
       re-opens it). Setting ``tempfile.tempdir`` in-process fixes that, because
       ``tempfile`` caches its choice in that module attribute.

    2. Directories created with mode **0700 are not writable**, while the same
       directory created with default permissions (0777 & ~umask) is. Since
       ``tempfile.mkdtemp`` always uses 0700, every temp directory it makes is
       unusable, and ``shutil.rmtree`` on it raises ``PermissionError``. Measured:

       ================================  ========
       mkdtemp (0700)                    DENIED
       makedirs default (0777 & ~umask)  OK
       makedirs mode=0700                DENIED
       ================================  ========

       So ``mkdtemp`` is replaced with a version that creates 0700 first and then
       widens it with ``os.chmod``, and ``TemporaryDirectory`` is replaced with
       one built on that plus a clean-up that never raises (leftovers are swept
       on the next run).
    """
    import shutil
    import tempfile

    cache = os.path.join(ROOT_DIR, ".cache", "tmp")
    os.makedirs(cache, exist_ok=True)  # default mode on purpose
    tempfile.tempdir = cache
    os.environ["TEMP"] = cache
    os.environ["TMP"] = cache

    _sweep_leftovers(cache)

    def _mkdtemp(suffix: str = "", prefix: str = "tmp", dir: str | None = None) -> str:
        """Create a uniquely named directory with default (writable) permissions.

        ``os.mkdir(path, 0o700)`` is not enough: the sandbox denies the follow-up
        ``os.chmod`` too, so the directory must never be created restrictive in
        the first place. ``os.makedirs`` with no mode argument uses
        0777 & ~umask, which is writable.
        """
        target = dir or tempfile.tempdir or cache
        os.makedirs(target, exist_ok=True)
        for _ in range(100):
            name = os.path.join(
                target, prefix + next(tempfile._get_candidate_names()) + suffix
            )
            try:
                os.makedirs(name)  # default mode: writable
            except FileExistsError:
                continue
            return name
        raise FileExistsError("could not create a unique temporary directory")

    class _ResilientTemporaryDirectory:
        """``tempfile.TemporaryDirectory`` that survives the sandbox's ACLs."""

        def __init__(self, suffix: str = "", prefix: str = "tmp", dir: str | None = None):
            self._closed = True  # set before mkdtemp so __del__ is safe on failure
            self.name = _mkdtemp(suffix, prefix, dir)
            self._closed = False

        def cleanup(self) -> None:
            if self._closed:
                return
            self._closed = True
            shutil.rmtree(self.name, ignore_errors=True)

        def __enter__(self) -> str:
            return self.name

        def __exit__(self, *exc_info) -> bool:
            self.cleanup()
            return False

        def __del__(self) -> None:
            self.cleanup()

    tempfile.mkdtemp = _mkdtemp  # type: ignore[assignment]
    tempfile.TemporaryDirectory = _ResilientTemporaryDirectory  # type: ignore[assignment]

    # matplotlib writes a font-cache lock file under the platform temp dir and
    # fails on the sandboxed one, so give it a cache inside the repo. Must be
    # set before matplotlib is imported anywhere.
    mpl_cache = os.path.join(ROOT_DIR, ".cache", "matplotlib")
    os.makedirs(mpl_cache, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache)
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(ROOT_DIR, ".cache"))

    return cache


def _sweep_leftovers(cache: str) -> None:
    """Best-effort removal of temp dirs a previous run could not delete."""
    import shutil

    try:
        entries = os.listdir(cache)
    except OSError:
        return
    for name in entries:
        if not name.startswith(("pre.quant.", "ort.quant.", "tmp")):
            continue
        shutil.rmtree(os.path.join(cache, name), ignore_errors=True)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened tensors.

    1.0 means the direction is preserved exactly; values far below 1.0 mean the
    quantized layer has rotated the feature vector, which is what kills
    accuracy in a classifier.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def sqnr(reference: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio in dB.

    SQNR = 10 * log10( sum(ref^2) / sum((test - ref)^2) )

    Higher is better. A well-behaved INT8 layer typically lands around
    30-40 dB; a layer that has collapsed drops below ~15 dB.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    err = np.asarray(test, dtype=np.float64).ravel() - ref
    signal = float(np.sum(ref * ref))
    noise = float(np.sum(err * err))
    if noise == 0.0:
        return float("inf")
    if signal == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(signal / noise))


def rel_l2(reference: np.ndarray, test: np.ndarray) -> float:
    """Relative L2 error ||test - ref|| / ||ref||, as a percentage-friendly ratio."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    err = np.asarray(test, dtype=np.float64).ravel() - ref
    norm = np.linalg.norm(ref)
    if norm == 0.0:
        return float("nan")
    return float(np.linalg.norm(err) / norm)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def dynamic_range(x: np.ndarray) -> dict:
    """Amplitude statistics used to explain *why* a layer quantizes badly."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {"min": float("nan"), "max": float("nan"), "absmax": float("nan"),
                "std": float("nan"), "kurtosis": float("nan")}
    mean = float(np.mean(x))
    std = float(np.std(x))
    centered = x - mean
    m4 = float(np.mean(centered ** 4))
    kurt = m4 / (std ** 4) if std > 0 else float("nan")
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "absmax": float(np.max(np.abs(x))),
        "std": std,
        "kurtosis": kurt,
    }


# --------------------------------------------------------------------------
# graph instrumentation
# --------------------------------------------------------------------------

#: Layer types whose output is worth comparing layer-by-layer.
DEFAULT_LAYER_OPS = (
    "Conv",
    "ConvTranspose",
    "MatMul",
    "Gemm",
    "Add",
    "Mul",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "Softmax",
    "Concat",
    "Resize",
)


def get_initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Best-effort shape inference; returns the model unchanged when it fails."""
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        return inferred
    except Exception:  # noqa: BLE001
        return model


def candidate_tensors(model_path: str, include_ops: Iterable[str] = DEFAULT_LAYER_OPS) -> list[dict]:
    """All tensors in a graph that are worth comparing, in execution order.

    Unlike :func:`instrument_graph` this does **not** modify the graph; it just
    enumerates candidates so they can be intersected with another graph first.

    Ordering is depth-first from the graph inputs following producer/consumer
    links, which approximates execution order far better than raw node order.
    """
    model = onnx.load(model_path)
    graph = model.graph

    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node

    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    include_ops = set(include_ops)
    seen: dict[str, int] = {}
    order = 0

    def visit(tensor: str) -> None:
        nonlocal order
        if tensor in seen or tensor not in producer:
            return
        node = producer[tensor]
        if node.op_type in include_ops:
            seen[tensor] = order
            order += 1
        for out in node.output:
            if out:
                for child in consumers.get(out, []):
                    for child_out in child.output:
                        if child_out:
                            visit(child_out)

    for node in graph.node:
        for inp in node.input:
            visit(inp)
    for node in graph.node:
        for out in node.output:
            if out and out not in seen and out not in {o.name for o in graph.output}:
                visit(out)

    out: list[dict] = []
    for tensor, index in sorted(seen.items(), key=lambda kv: kv[1]):
        node = producer[tensor]
        out.append(
            {
                "index": index,
                "tensor": tensor,
                "op_type": node.op_type,
                "name": node.name or "",
            }
        )
    return out


def intersect_tensors(
    reference_path: str,
    other_paths: Iterable[str],
    include_ops: Iterable[str] = DEFAULT_LAYER_OPS,
) -> list[dict]:
    """Candidates from `reference_path` that also exist in every other graph.

    Why not just take every Conv output? Because that silently loses layers.
    Measured on MobileNetV2 (FP32 vs a QDQ INT8 variant):

    ===========================================================  ======
    selection method                                             count
    ===========================================================  ======
    Conv outputs whose name also exists in the INT8 graph         17
    all layer candidates intersected by tensor name               29
    ===========================================================  ======

    The extra 12 are **10 Add (residual connections), 1 GlobalAveragePool
    and 1 Gemm (the classifier head)** - and residual Adds are among the
    most quantization-sensitive tensors in this model, so dropping them
    would hide part of the story.

    An earlier hypothesis was that the 17 shared names pointed at
    *different* computations in the two graphs (a pre-activation tensor
    matched against a post-activation one). Measurement disproved that:
    those 17 tensors give byte-identical cosine/SQNR values under both
    selection methods, so they are semantically the same tensor. The real
    defect is incomplete coverage, not mismatched comparison.

    Returns a list of layer dicts in execution order.
    """
    candidates = candidate_tensors(reference_path, include_ops)
    others = [set(_all_tensor_names(p)) for p in other_paths]
    return [c for c in candidates if all(c["tensor"] in o for o in others)]


def _all_tensor_names(model_path: str) -> list[str]:
    model = onnx.load(model_path)
    names: list[str] = []
    for node in model.graph.node:
        names.extend(n for n in node.output if n)
    return names


def instrument_explicit(
    model_path: str,
    out_path: str,
    tensors: Iterable[str],
) -> dict:
    """Rewrite a model so an explicit set of intermediate tensors becomes outputs.

    Returns ``{"layers": [...]}`` metadata in the same shape as
    :func:`instrument_graph`, and stores it under the ``dsh_layers`` metadata
    key so the mapping travels with the file.
    """
    wanted = list(dict.fromkeys(tensors))  # dedupe, preserve order
    model = onnx.load(model_path)
    model = _infer_shapes(model)

    available = _all_tensor_names(model_path)
    present = set(available) | {o.name for o in model.graph.output}
    missing = [t for t in wanted if t not in present]
    usable = [t for t in wanted if t in present]

    if not usable:
        raise RuntimeError(
            "none of the requested tensors exist in the graph "
            f"(requested {len(wanted)}, e.g. {wanted[:3]})"
        )

    op_of: dict[str, str] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                op_of.setdefault(out, node.op_type)

    known = {vi.name: vi for vi in list(model.graph.value_info) + list(model.graph.output)}
    existing = {o.name for o in model.graph.output}

    layers: list[dict] = []
    for index, tensor in enumerate(usable):
        layers.append(
            {
                "index": index,
                "tensor": tensor,
                "op_type": op_of.get(tensor, "?"),
                "name": "",
            }
        )
        if tensor in existing:
            continue
        vi = known.get(tensor)
        if vi is None:
            vi = helper.make_empty_tensor_value_info(tensor)
        model.graph.output.append(vi)

    meta = {
        "layers": layers,
        "source": os.path.basename(model_path),
        "missing": missing,
    }
    entry = model.metadata_props.add()
    entry.key = "dsh_layers"
    entry.value = json.dumps(meta)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    onnx.save(model, out_path)
    return meta


def instrument_graph(
    model_path: str,
    out_path: str,
    include_ops: Iterable[str] = DEFAULT_LAYER_OPS,
) -> dict:
    """Rewrite a model so intermediate activations become graph outputs.

    Returns a metadata dict::

        {"layers": [{"tensor": ..., "op_type": ..., "name": ..., "index": n}]}

    The metadata is also stored in the model's `metadata_props` under
    ``dsh_layers`` so the mapping travels with the file.
    """
    include_ops = set(include_ops)
    model = onnx.load(model_path)
    model = _infer_shapes(model)

    try:
        inferred_shapes = {
            vi.name: [
                d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in vi.type.tensor_type.shape.dim
            ]
            for vi in list(model.graph.value_info) + list(model.graph.output)
        }
    except Exception:  # noqa: BLE001
        inferred_shapes = {}

    existing_outputs = {o.name for o in model.graph.output}
    layers: list[dict] = []
    index = 0
    for node in model.graph.node:
        if node.op_type not in include_ops:
            continue
        for out_name in node.output:
            if not out_name or out_name in existing_outputs:
                continue
            layers.append(
                {
                    "index": index,
                    "tensor": out_name,
                    "op_type": node.op_type,
                    "name": node.name or "",
                    "shape": inferred_shapes.get(out_name),
                }
            )
            index += 1

    if not layers:
        raise RuntimeError(
            "no instrumentable layers found - check the op type list and that "
            "shape inference produced value_info"
        )

    # Build or reuse ValueInfoProto for each new output.
    known: dict[str, onnx.ValueInfoProto] = {
        vi.name: vi for vi in list(model.graph.value_info) + list(model.graph.output)
    }
    for layer in layers:
        tensor = layer["tensor"]
        vi = known.get(tensor)
        if vi is None:
            if layer["shape"]:
                dims = [
                    d if isinstance(d, int) else None for d in layer["shape"]
                ]
                vi = helper.make_tensor_value_info(
                    tensor, TensorProto.FLOAT, dims
                )
            else:
                vi = helper.make_empty_tensor_value_info(tensor)
        model.graph.output.append(vi)

    meta = {"layers": layers, "source": os.path.basename(model_path)}
    entry = model.metadata_props.add()
    entry.key = "dsh_layers"
    entry.value = json.dumps(meta)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    onnx.save(model, out_path)
    return meta


def load_layer_meta(model_path: str) -> dict:
    model = onnx.load(model_path)
    for prop in model.metadata_props:
        if prop.key == "dsh_layers":
            return json.loads(prop.value)
    raise KeyError(f"{model_path} has no dsh_layers metadata")


def extract_qparams(model_path: str) -> list[dict]:
    """Read every QuantizeLinear's scale / zero_point / axis.

    For per-tensor quantization `scale` is a scalar. For per-channel it is a
    1-D array with one entry per output channel, and `axis` says which
    dimension those channels live on.
    """
    model = onnx.load(model_path)
    inits = get_initializer_map(model)
    rows: list[dict] = []

    for node in model.graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        inputs = list(node.input)
        scale_name = inputs[1] if len(inputs) > 1 else None
        zp_name = inputs[2] if len(inputs) > 2 else None
        scale = inits.get(scale_name) if scale_name else None
        zp = inits.get(zp_name) if zp_name else None
        axis_attr = next(
            (a.i for a in node.attribute if a.name == "axis"), None
        )
        rows.append(
            {
                "node": node.name,
                "tensor": inputs[0] if inputs else None,
                "scale_name": scale_name,
                "per_channel": bool(scale is not None and np.size(scale) > 1),
                "axis": axis_attr,
                "num_scales": int(np.size(scale)) if scale is not None else 0,
                "scale_min": float(np.min(scale)) if scale is not None else float("nan"),
                "scale_max": float(np.max(scale)) if scale is not None else float("nan"),
                "scale_mean": float(np.mean(scale)) if scale is not None else float("nan"),
                "scale_std": float(np.std(scale)) if scale is not None else float("nan"),
                # per-channel scale spread is the headline reason per-channel wins
                "scale_spread": (
                    float(np.max(scale) / np.min(scale))
                    if scale is not None and np.size(scale) > 1 and np.min(scale) > 0
                    else 1.0
                ),
                "zero_point": float(np.mean(zp)) if zp is not None else float("nan"),
            }
        )
    return rows


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def effective_batch(session, requested: int) -> int:
    """Clamp a requested batch size to what the graph actually declares.

    The exported ONNX model has a fixed batch dimension, so feeding 4 images at
    once raises ``INVALID_ARGUMENT: Got: 4 Expected: 1``. Callers pass the batch
    they would like and get back one the graph accepts.
    """
    try:
        declared = session.get_inputs()[0].shape[0]
    except Exception:  # noqa: BLE001
        return requested
    if isinstance(declared, int) and declared > 0:
        return declared if declared != requested else requested
    return requested


def format_table(rows: list[dict], columns: list[str], title: str = "") -> str:
    """Render a list of dicts as a plain-text aligned table (markdown friendly)."""
    if not rows:
        return "(no rows)"
    widths = {}
    for col in columns:
        widths[col] = max(
            len(col),
            max(len(_fmt(r.get(col))) for r in rows),
        )
    lines = []
    if title:
        lines.append(title)
    header = "| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |"
    sep = "|" + "|".join("-" * (widths[col] + 2) for col in columns) + "|"
    lines.append(header)
    lines.append(sep)
    for row in rows:
        lines.append(
            "| " + " | ".join(_fmt(row.get(col)).ljust(widths[col]) for col in columns) + " |"
        )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        if abs(value) >= 1000 or (value != 0 and abs(value) < 1e-3):
            return "%.3e" % value
        return "%.4f" % value
    return str(value)


def write_csv(rows: list[dict], path: str) -> None:
    import csv

    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(obj: object, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")
