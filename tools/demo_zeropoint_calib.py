"""Why activations need a calibration set but weights do not.

Also derives zero_point by hand, which is the part most explanations skip.

Teaching material for the W1 wrap-up.
"""

import numpy as np


def symmetric_quantize(x, max_abs):
    """Symmetric: zero_point = 0, range [-127, 127]."""
    scale = max_abs / 127.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int16)
    return q, scale, 0


def asymmetric_quantize(x, vmin, vmax):
    """Asymmetric: zero_point shifts so real 0 maps to an integer inside range."""
    scale = (vmax - vmin) / 255.0
    # 从这里推导 zero_point：我们希望 vmin 映射到 -128
    #   q = v / scale + zero_point
    #   -128 = vmin / scale + zero_point
    #   zero_point = -128 - vmin / scale
    zero_point = int(round(-128.0 - vmin / scale))
    q = np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int16)
    return q, scale, zero_point


def main() -> None:
    print("=" * 78)
    print("第一部分：权重为什么不需要校准集")
    print("=" * 78)
    w = np.array([0.08, -0.12, 0.05, 8.40, -9.70, 6.30], dtype=np.float32)
    print("  一个卷积层的权重（训练完就固定，存在文件里）:")
    print("    ", w)
    print("  打开文件就能算出 max_abs = %.2f -> scale = %.6f" % (np.abs(w).max(), np.abs(w).max() / 127))
    print("  >>> 不需要任何数据，因为权重本身就是数据。")
    print()

    print("=" * 78)
    print("第二部分：激活为什么必须靠校准集")
    print("=" * 78)
    print("  激活 = 某一层的输出。它取决于输入图片。")
    print("  同一层，喂不同的图，输出范围完全不同:")
    print()
    rng = np.random.default_rng(0)
    for label, bias in [("一张暗图", 0.3), ("一张亮图", 4.0), ("一张高对比图", 8.0)]:
        act = np.abs(rng.standard_normal(1000) * bias).astype(np.float32)
        print("    %-12s 输出范围 [%.2f, %.2f]  max=%.2f"
              % (label, act.min(), act.max(), act.max()))
    print()
    print("  >>> 同一层，max 可能是 1.2，也可能是 24.0，差 20 倍。")
    print("      所以必须拿一批真实图片跑一遍，观察'实际会出现的范围'。")
    print("      这就是校准集的作用：探测分布，不训练任何东西。")
    print()

    print("=" * 78)
    print("第三部分：zero_point 怎么来的（手算）")
    print("=" * 78)
    act = np.array([0.0, 0.5, 1.2, 6.0], dtype=np.float32)
    print("  一个 ReLU 之后的激活值（全部非负）:")
    print("    ", act)
    print()

    # ---- 对称 ----
    print("  【对称量化】zero_point 强制为 0")
    q_s, scale_s, zp_s = symmetric_quantize(act, float(act.max()))
    print("    max_abs = %.2f" % act.max())
    print("    scale   = %.2f / 127 = %.6f" % (act.max(), scale_s))
    print("    zero_point = 0  （规定）")
    print("    %-8s -> 整数 %s" % ("原值", ""))
    for v, q in zip(act, q_s):
        print("      %-6.2f -> %6d" % (v, q))
    back_s = q_s * scale_s
    print("    反量化: ", np.round(back_s, 6))
    print("    误差  : ", np.round(np.abs(back_s - act), 6))
    print()

    # ---- 非对称 ----
    print("  【非对称量化】zero_point 把实数的 0 对齐到整数范围里")
    q_a, scale_a, zp_a = asymmetric_quantize(act, float(act.min()), float(act.max()))
    print("    min = %.2f, max = %.2f" % (act.min(), act.max()))
    print("    scale = (max - min) / 255 = %.6f" % scale_a)
    print("    推导 zero_point：我们要让 min(=0.00) 映射到整数 -128")
    print("      q = v/scale + zp")
    print("      -128 = %.2f/%.6f + zp" % (act.min(), scale_a))
    print("      zp   = -128 - %.6f = %d" % (act.min() / scale_a, zp_a))
    print()
    for v, q in zip(act, q_a):
        print("      %-6.2f -> %6d" % (v, q))
    back_a = (q_a - zp_a) * scale_a
    print("    反量化: ", np.round(back_a, 6))
    print("    误差  : ", np.round(np.abs(back_a - act), 6))
    print()

    print("=" * 78)
    print("对比结论")
    print("=" * 78)
    print("  对称  : 整数范围用了 [-127, 0, 127]，负数那一半（128格）全浪费")
    print("          误差最大 %.6f" % np.abs(back_s - act).max())
    print("  非对称: 整数范围用满了 [-128, 127]，256 格全给 [0, 6]")
    print("          误差最大 %.6f" % np.abs(back_a - act).max())
    print("  改善  : %.1f 倍" % (np.abs(back_s - act).max() / np.abs(back_a - act).max()))
    print()
    print("  >>> 这就是为什么 ReLU 后的激活要用非对称量化：")
    print("      值全是非负的，用对称等于把 256 格里的 128 格扔了。")
    print()
    print("  而权重通常正负都有、近似对称，所以用对称更省事（不用存 zero_point）。")


if __name__ == "__main__":
    main()
