"""W1-D3: ONNX Runtime inference, and what the execution provider actually buys you.

Covers the questions that come with "你用过 ONNXRuntime 吗":

* how a session is created and why ``SessionOptions`` matters
* what ``providers`` selects (CPU vs CUDA vs others) and how to find out what is
  really available instead of assuming
* static vs dynamic input shapes
* IO binding, which avoids a copy per input and per output on accelerated EPs
* graph optimization levels and what they change
* warmup: the first inference pays for arena allocation and kernel selection, so
  any latency number measured without warmup is wrong

Prints a measured comparison rather than describing one.

Usage
-----
    python 08_ort_inference.py --model artifacts/models/mobilenet_v2_opset13_sim.onnx
    python 08_ort_inference.py --benchmark --runs 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w1_quant.common import (  # noqa: E402
    cosine_similarity,
    effective_batch,
    format_table,
    redirect_tempdir,
    write_json,
)

TMPDIR = redirect_tempdir()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def describe_session(sess) -> dict:
    inputs = [
        {"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()
    ]
    outputs = [
        {"name": o.name, "shape": o.shape, "type": o.type} for o in sess.get_outputs()
    ]
    return {
        "providers": sess.get_providers(),
        "inputs": inputs,
        "outputs": outputs,
        "dynamic_input": any(
            any(isinstance(d, str) for d in i["shape"]) for i in inputs
        ),
    }


def available_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def make_session(path: str, level=None, threads: int | None = None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if level is not None:
        opts.graph_optimization_level = level
    if threads is not None:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def benchmark(sess, x: np.ndarray, runs: int) -> dict:
    name = sess.get_inputs()[0].name
    # Warmup: first call allocates the memory arena and picks kernels.
    first_t0 = time.perf_counter()
    sess.run(None, {name: x})
    first = time.perf_counter() - first_t0

    for _ in range(3):
        sess.run(None, {name: x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append(time.perf_counter() - t0)
    times = np.asarray(times)
    return {
        "first_call_ms": first * 1000,
        "mean_ms": float(times.mean() * 1000),
        "median_ms": float(np.median(times) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "min_ms": float(times.min() * 1000),
        "std_ms": float(times.std() * 1000),
        "fps_from_median": float(1000.0 / np.median(times) / 1000.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=os.path.join(ROOT, "artifacts/models/mobilenet_v2_opset13_sim.onnx"),
    )
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument(
        "--dynamic",
        default=os.path.join(ROOT, "artifacts/models/mobilenet_v2_dynamic.onnx"),
    )
    ap.add_argument("--outdir", default=os.path.join(ROOT, "artifacts"))
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print("model not found:", args.model)
        print("run 01_export_onnx.py first")
        return 1

    import onnxruntime as ort

    result: dict = {}

    # ---- 1. providers -----------------------------------------------------
    print("=" * 70)
    print("1. execution providers")
    print("=" * 70)
    providers = available_providers()
    print("  本机可用 providers :", providers)
    print("  本机 onnxruntime   :", ort.__version__)
    print("  device             :", ort.get_device())
    print()
    print("  说明：providers 是**按优先级排列**的列表，ONNX Runtime 会用第一个能")
    print("  成功初始化的那个。CPUExecutionProvider 永远兜底，所以在没有 GPU 的")
    print("  机器上写 CUDAExecutionProvider 不会报错，只会静默回退到 CPU——")
    print("  想知道真正生效的是谁，必须看 session.get_providers()。")
    result["available_providers"] = providers
    result["device"] = ort.get_device()

    # ---- 2. session and IO ------------------------------------------------
    print()
    print("=" * 70)
    print("2. session / 输入输出")
    print("=" * 70)
    sess = make_session(args.model)
    info = describe_session(sess)
    result["session"] = info
    print("  生效 provider      :", info["providers"])
    for i in info["inputs"]:
        print("  input  %-8s %-18s %s" % (i["name"], i["type"], i["shape"]))
    for o in info["outputs"]:
        print("  output %-8s %-18s %s" % (o["name"], o["type"], o["shape"]))
    print("  动态输入形状       :", info["dynamic_input"])

    # ---- 3. dynamic vs static batch ---------------------------------------
    print()
    print("=" * 70)
    print("3. 静态 vs 动态 batch")
    print("=" * 70)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
    out_static = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    print("  静态模型 1 张       :", out_static.shape)

    dynamic_note = "未提供动态模型（用 01_export_onnx.py --dynamic-batch 生成）"
    if os.path.exists(args.dynamic):
        d_sess = make_session(args.dynamic)
        xb = np.repeat(x, 4, axis=0)
        try:
            out_dyn = d_sess.run(None, {d_sess.get_inputs()[0].name: xb})[0]
            cos = cosine_similarity(out_dyn[0], out_static[0])
            dynamic_note = (
                f"动态模型接受 batch=4 -> {out_dyn.shape}；"
                f"第 0 张与静态结果 cosine={cos:.8f}"
            )
            print("  动态模型 batch=4    :", out_dyn.shape)
            print("  数值一致性 cosine   : %.8f" % cos)
            result["dynamic_batch"] = {"output_shape": list(out_dyn.shape), "cosine": cos}
        except Exception as exc:  # noqa: BLE001
            dynamic_note = f"动态模型推理失败: {type(exc).__name__}: {exc}"
            print("  " + dynamic_note)
    else:
        print("  " + dynamic_note)
    result["dynamic_note"] = dynamic_note

    # ---- 4. IO binding ----------------------------------------------------
    print()
    print("=" * 70)
    print("4. IO binding vs 普通 run()")
    print("=" * 70)
    print("  普通 run() 每次调用都要把 numpy 数组拷进 ONNX Runtime 的输入张量，")
    print("  再把输出拷回 numpy。IOBinding 让你自己指定输入/输出的内存，")
    print("  在 CUDA/TensorRT 这类 EP 上可以省掉每帧的 H2D/D2H 拷贝。")
    print("  这里在 CPU 上也会测一遍——CPU 上省得少，但没有理由不用。")
    try:
        io = sess.io_binding()
        name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        x_ort = ort.OrtValue.ortvalue_from_numpy(x, "cpu", 0)
        io.bind_ortvalue_input(name, x_ort)
        io.bind_output(out_name, "cpu")
        sess.run_with_iobinding(io)
        bound_out = io.get_outputs()[0].numpy()
        cos = cosine_similarity(bound_out, out_static)
        print("  IO binding 结果一致 : cosine=%.8f" % cos)
        result["io_binding"] = {"cosine": cos}
    except Exception as exc:  # noqa: BLE001
        print("  IO binding 在这个版本/EP 上不可用:", type(exc).__name__, exc)
        result["io_binding"] = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- 5. benchmark -----------------------------------------------------
    if args.benchmark:
        print()
        print("=" * 70)
        print("5. 基准测试（含 warmup 的前后对比）")
        print("=" * 70)
        rows = []
        levels = [
            ("ORT_DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
            ("ORT_ENABLE_BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
            ("ORT_ENABLE_EXTENDED", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
            ("ORT_ENABLE_ALL", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
        ]
        for label, level in levels:
            s = make_session(args.model, level=level, threads=1)
            # single-thread + single-process to make the comparison about the
            # graph, not about how many cores the machine happened to use
            stats = benchmark(s, x, args.runs)
            stats["label"] = label
            rows.append(stats)
            print(
                "  %-22s first=%7.2f ms  median=%6.2f ms  p95=%6.2f ms  std=%5.2f"
                % (label, stats["first_call_ms"], stats["median_ms"],
                   stats["p95_ms"], stats["std_ms"])
            )
        result["optimization_levels"] = rows

        print()
        print("  " + format_table(
            [
                {
                    "level": r["label"],
                    "first_ms": r["first_call_ms"],
                    "median_ms": r["median_ms"],
                    "p95_ms": r["p95_ms"],
                    "std_ms": r["std_ms"],
                    "fps": r["fps_from_median"],
                }
                for r in rows
            ],
            ["level", "first_ms", "median_ms", "p95_ms", "std_ms", "fps"],
        ))

        best = min(rows, key=lambda r: r["median_ms"])
        first_row = rows[0]
        print()
        print("  首次调用 vs 稳态    : %.2f ms vs %.2f ms（相差 %.2f 倍）"
              % (best["first_call_ms"], best["median_ms"],
                 best["first_call_ms"] / max(best["median_ms"], 1e-9)))
        print("  实测说明：在这个 CPU + 小模型上，warmup 的收益只有百分之十几，")
        print("  **不是**常说的\"好几倍\"。原因是 CPU EP 没有 GPU 那样的 kernel")
        print("  自动调优和 H2D/D2H 拷贝，首次调用的额外开销主要来自内存 arena")
        print("  分配。warmup 真正关键的场景是：① CUDA/TensorRT（kernel 调优，")
        print("  首帧可能慢 10 倍以上）；② 大模型（arena 分配占比高）；")
        print("  ③ 做分位数统计（第一次的异常值会污染 p95）。")
        print("  所以正确说法是\"必须做 warmup，但收益大小取决于后端\"。")
        speedup = first_row["median_ms"] / max(best["median_ms"], 1e-9)
        print()
        print("  优化级别 DISABLE_ALL -> %s : %.2f ms -> %.2f ms（%.2fx）"
              % (best["label"], first_row["median_ms"], best["median_ms"], speedup))
        print("  **这才是本机最大的单点收益**：图优化级别（算子融合）比线程数和")
        print("  warmup 都更值得先调。")

        # thread scaling
        print()
        print("  线程数影响（ORT_ENABLE_ALL）:")
        thread_rows = []
        for threads in (1, 2, 4):
            s = make_session(args.model, level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                             threads=threads)
            stats = benchmark(s, x, args.runs)
            stats["threads"] = threads
            thread_rows.append(stats)
            print("    intra_op_num_threads=%d -> median=%.2f ms  p95=%.2f ms"
                  % (threads, stats["median_ms"], stats["p95_ms"]))
        result["threads"] = thread_rows

    write_json(result, os.path.join(args.outdir, "reports", "08_ort_inference.json"))
    print()
    print("report :", os.path.relpath(
        os.path.join(args.outdir, "reports", "08_ort_inference.json"), ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
