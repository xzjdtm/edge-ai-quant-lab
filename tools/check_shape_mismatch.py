"""Check whether shape mismatches cause layers to be silently skipped.

`compare_models` skips any tensor whose FP32 and INT8 shapes differ. If that
happens a lot, the analysis is quietly comparing fewer layers than it claims.
"""

import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import onnx


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fp32 = os.path.join(root, "artifacts/models/mobilenet_v2_opset13_sim_preproc.onnx")
    int8 = os.path.join(root, "artifacts/models/mobilenet_v2_opset13_sim_qdq_perchannel_minmax.onnx")

    print("=" * 74)
    print("检查：FP32 图和 INT8 图里，同名张量的形状是否一致")
    print("=" * 74)

    def shapes(path):
        m = onnx.load(path)
        out = {}
        for node in m.graph.node:
            for o in node.output:
                if o:
                    out[o] = None
        # 用 value_info 补形状
        for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input):
            dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                    for d in vi.type.tensor_type.shape.dim]
            out[vi.name] = dims
        return out

    s32, s8 = shapes(fp32), shapes(int8)
    common = sorted(set(s32) & set(s8))
    same = [n for n in common if s32[n] == s8[n]]
    diff = [n for n in common if s32[n] != s8[n]]
    unknown = [n for n in common if s32[n] is None or s8[n] is None]

    print("  两边都存在的张量 : %d" % len(common))
    print("  形状相同         : %d" % len(same))
    print("  形状不同         : %d" % len(diff))
    print("  形状未知         : %d" % len(unknown))
    if diff:
        print()
        print("  形状不同的具体是哪些:")
        for n in diff[:10]:
            print("    %-50s FP32=%s  INT8=%s" % (n[-48:], s32[n], s8[n]))

    # 从实际结果 CSV 反推：到底比了多少层
    print()
    print("=" * 74)
    print("实际结果里比了多少层")
    print("=" * 74)
    csv_path = os.path.join(root, "artifacts/tables/04_layer_metrics.csv")
    if os.path.exists(csv_path):
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
        for variant in sorted({r["variant"] for r in rows}):
            n = sum(1 for r in rows if r["variant"] == variant)
            print("  %-28s 比了 %d 层" % (variant, n))
    else:
        print("  (没找到 04_layer_metrics.csv)")

    # 演示形状不同时相减会发生什么
    print()
    print("=" * 74)
    print("演示：形状不同时相减会怎样（这是 continue 在防的事）")
    print("=" * 74)
    a = np.array([[[1.0, 2.0, 3.0]]])          # (1, 1, 3)
    b = np.array([[1.0, 2.0, 3.0]])            # (1, 3)
    print("  a 形状 %s  值 %s" % (a.shape, a.ravel()))
    print("  b 形状 %s  值 %s" % (b.shape, b.ravel()))
    print()
    try:
        d = a - b
        print("  a - b 形状 = %s  值 =" % (d.shape,))
        print(d)
        print()
        print("  >>> 没有报错！但形状从 (1,1,3) 变成了 %s。" % (d.shape,))
        print("      numpy 把 b 广播成了 (1,1,3)。这里恰好每层都一样所以结果没变，")
        print("      但真实情况下 b 的 (1,3) 会被复制 1 次铺开，语义可能是错的。")
    except ValueError as exc:
        print("  报错:", exc)

    print()
    print("  对比：如果形状完全无法广播，numpy 会报错（这是好事）:")
    x = np.zeros((2, 3))
    y = np.zeros((4, 5))
    try:
        x - y
    except ValueError as exc:
        print("    (2,3) - (4,5) ->", exc)
    print()
    print("  所以最危险的是'能广播但语义错'的情况：不报错，结果却是错的。")


if __name__ == "__main__":
    main()
