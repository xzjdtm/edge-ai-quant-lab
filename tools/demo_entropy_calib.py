"""How Entropy (KL-divergence) calibration picks a clipping threshold.

The problem it solves: a few extreme outliers in the activation distribution
stretch the scale so far that the bulk of values loses resolution. Clipping the
tails trades a little range for much better resolution where the mass actually
is.

MinMax uses the full observed range. Entropy searches for a threshold T that
minimizes the KL divergence between the original distribution and the quantized
one.
"""

import numpy as np


def quantize_with_clip(x, clip):
    """Quantize x into [-128,127] using symmetric range [-clip, +clip]."""
    scale = clip / 127.0
    q = np.clip(np.round(x / scale), -128, 127)
    return q, scale


def dequantize(q, scale):
    return q * scale


def kl_divergence(p, q):
    """KL(p || q); p and q are histograms normalised to sum to 1."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))))


def main() -> None:
    rng = np.random.default_rng(0)

    # 主体是 N(0, 1)，但混入了 0.5% 的离群值（模拟 ReLU 后的少数大激活）
    body = rng.standard_normal(10_000).astype(np.float32)
    outliers = rng.uniform(8.0, 15.0, size=50).astype(np.float32)
    x = np.concatenate([body, outliers])

    print("=" * 78)
    print("激活值分布")
    print("=" * 78)
    print("  样本数        : %d" % x.size)
    print("  主体           : 正态分布 N(0,1)")
    print("  离群值         : %d 个 (%.1f%%), 范围 [%.1f, %.1f]"
          % (outliers.size, outliers.size / x.size * 100, outliers.min(), outliers.max()))
    print("  absmax        : %.2f" % np.abs(x).max())
    print("  99.5%% 分位数 : %.2f" % np.percentile(np.abs(x), 99.5))
    print()
    print("  >>> 注意：absmax = %.1f，但 99.5%% 的值都在 %.2f 以内。"
          % (np.abs(x).max(), np.percentile(np.abs(x), 99.5)))
    print("      用 absmax 当 clip 范围，那 10000 个正常值只占整数范围的 1/15。")
    print()

    # 原始分布的直方图（当作"真值"）
    bins = 256
    lo, hi = -x.max(), x.max()
    ref_hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    ref_hist = ref_hist / ref_hist.sum()

    print("=" * 78)
    print("比较不同 clip 阈值：误差 vs 信息损失")
    print("=" * 78)
    print("  %-12s %-10s %-14s %-12s %-12s" % ("clip 阈值", "被截断%", "MSE", "KL 散度", "说明"))
    print("  " + "-" * 74)

    results = []
    for clip in [np.abs(x).max(), 8.0, 5.0, 3.0, 2.5, 2.0, 1.5, 1.0]:
        q, scale = quantize_with_clip(x, clip)
        back = dequantize(q, scale)
        mse = float(np.mean((back - x) ** 2))
        clipped_pct = float(np.mean(np.abs(x) > clip) * 100)

        # 量化后的分布
        q_hist, _ = np.histogram(back, bins=bins, range=(lo, hi))
        q_hist = q_hist / max(q_hist.sum(), 1)
        kl = kl_divergence(ref_hist, q_hist)

        note = ""
        if clip == np.abs(x).max():
            note = "<- MinMax（不截断）"
        results.append((clip, clipped_pct, mse, kl, note))
        print("  %-12.2f %-10.2f %-14.6f %-12.4f %s" % (clip, clipped_pct, mse, kl, note))

    print()
    best_mse = min(results, key=lambda r: r[2])
    best_kl = min(results, key=lambda r: r[3])
    print("  MSE 最小的 clip   : %.2f  (MSE=%.6f)" % (best_mse[0], best_mse[2]))
    print("  KL  最小的 clip   : %.2f  (KL=%.4f)" % (best_kl[0], best_kl[3]))
    print()
    print("=" * 78)
    print("Entropy 校准在做什么")
    print("=" * 78)
    print("  它在很多 clip 阈值上试探，对每个阈值计算:")
    print("    '量化后的分布' vs '原始分布' 的 KL 散度")
    print("  然后选 KL 最小的那个阈值。")
    print()
    print("  KL 散度 = 用 q 去近似 p 时损失的信息量（单位是比特）")
    print("    KL=0    -> 两个分布完全相同，没有信息损失")
    print("    KL 变大 -> 量化后的分布偏离原始分布")
    print()
    print("  >>> 为什么'截断尾部'反而可能更好：")
    print("      不截断时，为了覆盖 max=%.1f，scale 被撑大，10000 个正常值")
    print("      挤在很少的几个整数格子里，分布严重失真（KL 大）。" % np.abs(x).max())
    print("      截断后，正常的 99.5%% 值能用更多格子表示，分布更接近原始。")
    print("      被截断的 0.5%% 离群值变成 saturate 值，但它们是少数。")
    print()
    print("  三种校准方法的区别：")
    print("    MinMax     : 直接取 [min, max]，不截断（对离群值敏感）")
    print("    Entropy    : 搜索使 KL 散度最小的阈值（本脚本演示的就是这个）")
    print("    Percentile : 直接砍掉固定比例（如 99.999%）的尾部")


if __name__ == "__main__":
    main()
