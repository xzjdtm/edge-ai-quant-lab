"""W1-D2: export a torchvision classifier to ONNX, then simplify it.

Answers the interview question "导出 ONNX 踩过什么坑" with measured numbers
instead of hand-waving:

* what `opset_version` changes in the graph
* what `dynamic_axes` actually buys you
* what `do_constant_folding` folds away
* how many nodes `onnxsim` removes and why

Usage
-----
    python 01_export_onnx.py --model mobilenet_v2 --opset 13
    python 01_export_onnx.py --model squeezenet1_1 --dynamic-batch
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w1_quant.common import redirect_tempdir, write_json  # noqa: E402

TMPDIR = redirect_tempdir()

MODELS = {
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights"),
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "squeezenet1_1": ("squeezenet1_1", "SqueezeNet1_1_Weights"),
    "shufflenet_v2_x0_5": ("shufflenet_v2_x0_5", "ShuffleNet_V2_X0_5_Weights"),
}


def graph_stats(onnx_path: str) -> dict:
    import onnx

    model = onnx.load(onnx_path)
    ops: dict[str, int] = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    size = os.path.getsize(onnx_path)
    return {
        "path": os.path.basename(onnx_path),
        "size_mb": round(size / 1e6, 2),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
        "has_dynamic_batch": any(
            (d.dim_param or "") != ""
            for i in model.graph.input
            for d in i.type.tensor_type.shape.dim
        ),
        "top_ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])[:12]),
        "ir_version": model.ir_version,
        "opset": {o.domain or "ai.onnx": o.version for o in model.opset_import},
        "metadata": {p.key: p.value for p in model.metadata_props},
    }


def build_model(name: str):
    import torch
    from torchvision import models

    factory_name, weights_name = MODELS[name]
    weights = getattr(models, weights_name).DEFAULT
    model = getattr(models, factory_name)(weights=weights)
    model.eval()
    return model, weights


def use_workspace_cache() -> None:
    """Keep torch/torchvision downloads inside the repo.

    The default `~/.cache/torch` is outside the sandbox's writable area, so
    pretrained-weight downloads fail with a permission error unless we redirect
    them.
    """
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache = os.path.join(here, ".cache", "torch")
    os.makedirs(cache, exist_ok=True)
    os.environ["TORCH_HOME"] = cache
    # torch.hub reads XDG_CACHE_HOME on Linux and TORCH_HOME everywhere.
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(here, ".cache"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet_v2", choices=sorted(MODELS))
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--outdir", default="artifacts/models")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--dynamic-batch", action="store_true")
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument(
        "--no-constant-folding",
        action="store_true",
        help="export with do_constant_folding=False to see what it would have folded",
    )
    ap.add_argument("--report", default="artifacts/reports/01_export_report.json")
    args = ap.parse_args()

    import torch

    use_workspace_cache()
    os.makedirs(args.outdir, exist_ok=True)
    model, weights = build_model(args.model)
    print(f"model      : {args.model} (pretrained={weights})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters : {n_params/1e6:.2f} M")

    torch.manual_seed(0)
    dummy = torch.randn(1, 3, args.img_size, args.img_size)

    # Sanity check the PyTorch reference output before exporting anything.
    with torch.no_grad():
        ref = model(dummy).numpy()
    print(f"torch out  : {ref.shape}, argmax={int(np.argmax(ref))}, "
          f"sum={float(ref.sum()):.4f}")

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    onnx_path = os.path.join(args.outdir, f"{args.model}_opset{args.opset}.onnx")
    export_kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=not args.no_constant_folding,
    )
    with torch.no_grad():
        torch.onnx.export(model, dummy, onnx_path, **export_kwargs)
    print(f"exported   : {onnx_path}")

    report: dict = {
        "model": args.model,
        "params_m": round(n_params / 1e6, 3),
        "opset_requested": args.opset,
        "dynamic_batch": args.dynamic_batch,
        "do_constant_folding": not args.no_constant_folding,
        "torch_output": {
            "argmax": int(np.argmax(ref)),
            "sum": float(ref.sum()),
            "std": float(ref.std()),
        },
        "stages": {},
    }

    # ---------------------------------------------------------------- verify
    import onnx

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported)
    print("onnx check : OK")
    report["stages"]["exported"] = graph_stats(onnx_path)

    # ------------------------------------------------------------ onnxsim
    if not args.no_simplify:
        import onnxsim

        simplified, ok = onnxsim.simplify(exported)
        if not ok:
            print("onnxsim    : check FAILED, keeping unsimplified graph")
            report["simplify_ok"] = False
        else:
            sim_path = os.path.join(args.outdir, f"{args.model}_opset{args.opset}_sim.onnx")
            onnx.save(simplified, sim_path)
            print(f"simplified : {sim_path}")
            report["simplify_ok"] = True
            report["stages"]["simplified"] = graph_stats(sim_path)

            # Verify the simplified graph actually computes the same thing.
            import onnxruntime as ort

            sess = ort.InferenceSession(sim_path, providers=["CPUExecutionProvider"])
            got = sess.run(None, {"input": dummy.numpy()})[0]
            diff = float(np.max(np.abs(got - ref)))
            cos = float(
                np.dot(got.ravel(), ref.ravel())
                / (np.linalg.norm(got) * np.linalg.norm(ref))
            )
            print(f"verify     : max|diff|={diff:.3e}  cosine={cos:.8f}")
            report["verify"] = {"max_abs_diff": diff, "cosine": cos}
            report["stages"]["simplified"]["verified"] = bool(diff < 1e-4)
    else:
        report["simplify_ok"] = None

    write_json(report, args.report)
    print(f"\nreport     : {args.report}")

    # Human-readable comparison table.
    print("\n=== 导出前后对比 ===")
    for stage, stats in report["stages"].items():
        print(
            f"  {stage:12s} nodes={stats['nodes']:4d}  "
            f"initializers={stats['initializers']:4d}  "
            f"size={stats['size_mb']:7.2f} MB  "
            f"dynamic_batch={stats['has_dynamic_batch']}"
        )
    if "simplified" in report["stages"]:
        before = report["stages"]["exported"]["nodes"]
        after = report["stages"]["simplified"]["nodes"]
        print(f"  onnxsim 删除了 {before - after} 个节点 "
              f"({(before - after) / before * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
