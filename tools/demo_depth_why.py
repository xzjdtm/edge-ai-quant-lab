"""Why does quantization error grow with depth in MobileNetV2?

Checks two candidate explanations against the measured per-layer data:
  1. error accumulates: each layer inherits the error of its input
  2. depthwise convolutions are the fragile ones

Usage:  python demo_depth_why.py
"""

from __future__ import annotations

import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = list(csv.DictReader(open(os.path.join(root, "artifacts/tables/04_layer_metrics.csv"),
                                   encoding="utf-8-sig")))

    for variant in ("qdq_perchannel_minmax", "qdq_pertensor_minmax"):
        sub = [r for r in rows if r["variant"] == variant]
        sub.sort(key=lambda r: int(r["order"]))
        print("=" * 78)
        print("变体: %s" % variant)
        print("=" * 78)

        order = np.array([int(r["order"]) for r in sub], dtype=float)
        cos = np.array([float(r["cosine"]) for r in sub])

        # 1) 误差是否随层深累积
        r = float(np.corrcoef(order, cos)[0, 1])
        print("  1) 层序号 vs cosine : 相关系数 = %+.3f" % r)
        if r < -0.5:
            print("     结论：显著负相关 -> 越靠后越差，误差确实随层深累积")
        elif r > 0.5:
            print("     结论：显著正相关 -> 越靠后越好（不符合预期）")
        else:
            print("     结论：关系弱 -> 误差不是简单随层深线性累积")

        # 2) 前段 / 中段 / 后段 的平均值
        n = len(sub)
        for label, sl in (("前 1/3", slice(0, n // 3)),
                          ("中 1/3", slice(n // 3, 2 * n // 3)),
                          ("后 1/3", slice(2 * n // 3, n))):
            part = cos[sl]
            print("     %-8s mean cosine = %.6f   min = %.6f" % (label, part.mean(), part.min()))
        print()

        # 3) 按算子类型分组统计
        print("  2) 按算子类型看误差（哪种算子最脆弱）")
        kinds: dict[str, list[float]] = {}
        for row in sub:
            t = row["tensor"]
            op = row["op_type"]
            # 区分 depthwise conv（MobileNet 里是 conv.1 那种）
            if op == "Conv":
                kind = "depthwise Conv" if re.search(r"conv\.1/", t) else "普通 Conv"
            else:
                kind = op
            kinds.setdefault(kind, []).append(float(row["cosine"]))
        print("     %-16s %6s %12s %12s" % ("类型", "层数", "平均cosine", "最差cosine"))
        for kind in sorted(kinds, key=lambda k: np.mean(kinds[k])):
            v = kinds[kind]
            print("     %-16s %6d %12.6f %12.6f" % (kind, len(v), np.mean(v), np.min(v)))
        print()

    print("=" * 78)
    print("怎么读这些数字")
    print("=" * 78)
    print("  如果 depthwise Conv 的平均 cosine 明显低于普通 Conv，")
    print("  说明'逐深度卷积脆弱'这个解释成立 —— 这也和 literature 一致：")
    print("  逐深度卷积每个输出通道只有一个卷积核，通道间动态范围差异大，")
    print("  per-tensor 的单一 scale 无法同时照顾所有通道。")


if __name__ == "__main__":
    main()
