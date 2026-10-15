"""Hand-calculate the FP32 -> INT8 conversion step by step.

Used as teaching material for LAB lesson 4. Shows that quantization is not just
"fewer bytes": it changes the number representation AND discards precision, and
both effects can be computed by hand.
"""

import numpy as np


def main() -> None:
    print("=" * 78)
    print("手算全过程：6 个数值怎么变成 INT8 整数")
    print("=" * 78)

    vals = np.array([0.08, -0.12, 0.05, 8.40, -9.70, 6.30], dtype=np.float32)
    print("原始 FP32 权重：%d 个数 × 4 字节 = %d 字节" % (vals.size, vals.nbytes))
    print("  ", vals)
    print()

    max_abs = float(np.abs(vals).max())
    print("【第 1 步】找最大绝对值")
    for v in vals:
        print("  |%6.2f| = %.2f" % (v, abs(v)))
    print("  max_abs = %.2f" % max_abs)
    print()

    scale = max_abs / 127.0
    print("【第 2 步】算 scale：把 max_abs 映射到整数 127")
    print("  scale = %.2f / 127 = %.6f" % (max_abs, scale))
    print("  含义：1 个整数刻度 = %.6f" % scale)
    print()

    q_float = vals / scale
    q_int = np.round(q_float).astype(np.int8)
    print("【第 3 步】每个数除以 scale，然后四舍五入")
    print("  %-8s %-14s %-14s %s" % ("原值", "除以scale", "四舍五入", "得到整数"))
    for v, qf, qi in zip(vals, q_float, q_int):
        print("  %-8.2f %-14.4f %-14.4f %d" % (v, qf, round(float(qf)), qi))
    print()
    print("  现在这些是 INT8 整数，每个 1 字节 = %d 字节" % q_int.nbytes)
    print()

    dq = q_int.astype(np.float32) * scale
    print("【第 4 步】反量化回去（用同一个 scale 乘）")
    print("  %-8s %-8s %-14s %-12s %s" % ("原值", "整数", "反量化值", "绝对误差", "相对误差"))
    worst_rel = 0.0
    for v, qi, d in zip(vals, q_int, dq):
        err = abs(float(d) - float(v))
        rel = err / abs(float(v)) * 100 if v != 0 else 0.0
        worst_rel = max(worst_rel, rel)
        print("  %-8.2f %-8d %-14.6f %-12.6f %.1f%%" % (v, qi, d, err, rel))
    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print("  字节数 : %d -> %d 字节   （这就是 13.95 MB -> 3.49 MB 的原因）"
          % (vals.nbytes, q_int.nbytes))
    print("  代价   : 小数值的相对误差最高 %.0f%%" % worst_rel)
    print()
    print("  >>> 所以 FP32 -> INT8 是两件事合在一起：")
    print("      ① 表示方式变了：浮点数 -> 定点整数，字节数变成 1/4")
    print("      ② 精度丢了：小数值被舍入，这是必须付出的代价")
    print()
    print("  per-channel 就是用来压制 ② 的：")
    print("      把小数值的通道单独给一把小 scale，让它也能用满整数范围。")


if __name__ == "__main__":
    main()
