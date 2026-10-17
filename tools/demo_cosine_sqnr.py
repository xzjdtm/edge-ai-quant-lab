"""Explain cosine similarity and SQNR with visual, concrete examples.

Both metrics appear everywhere in this project; this makes their behaviour
intuitive instead of formula-recitation.

Teaching material for W1 wrap-up.
"""

import numpy as np


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sqnr(ref, test):
    ref = np.asarray(ref, dtype=np.float64).ravel()
    err = np.asarray(test, dtype=np.float64).ravel() - ref
    return float(10 * np.log10(np.sum(ref ** 2) / np.sum(err ** 2)))


def main() -> None:
    print("=" * 78)
    print("一、cosine 相似度：衡量'方向是否一致'，与长度无关")
    print("=" * 78)

    a = np.array([1.0, 2.0, 3.0])
    cases = [
        ("完全一样", a),
        ("全部放大 100 倍", a * 100),
        ("全部缩小到 1/100", a / 100),
        ("全部取负", -a),
        ("只改一个元素 (3->4)", np.array([1.0, 2.0, 4.0])),
        ("完全垂直", np.array([3.0, -1.0, -0.3333])),
        ("方向完全不同", np.array([-1.0, 0.0, 0.0])),
    ]
    print("  参照向量 a = %s" % a.tolist())
    print()
    print("  %-26s %-22s %10s" % ("情况", "对比向量", "cosine"))
    for label, b in cases:
        print("  %-26s %-22s %10.6f" % (label, np.round(b, 3).tolist(), cosine(a, b)))
    print()
    print("  关键观察：")
    print("    - 放大/缩小 100 倍，cosine 仍然是 1.0  -> 对'长度'完全不敏感")
    print("    - 取负，cosine = -1.0                    -> 方向完全相反")
    print("    - 垂直，cosine = 0                       -> 毫无关系")
    print()
    print("  公式: cos(a,b) = (a·b) / (|a| * |b|)")
    print("    分子 a·b = 两个向量的点积（对应元素相乘再求和）")
    print("    分母 |a|*|b| = 两个向量的长度相乘")
    print("  分母的作用是'归一化'——把长度除掉，只留方向。")
    print()
    print("  >>> 为什么量化分析用它：神经网络每层的输出是一个高维向量。")
    print("      cosine 接近 1 说明'特征方向没变'，即使整体幅度有偏移。")
    print("      cosine 掉到 0.9 说明方向已经偏了 25 度左右，下游会受影响。")
    print()

    print("=" * 78)
    print("二、SQNR：信噪比，衡量'信号能量 / 噪声能量'")
    print("=" * 78)
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    print("  参照 signal = %s，能量 sum(ref^2) = %.1f" % (ref.tolist(), np.sum(ref ** 2)))
    print()
    print("  %-30s %-20s %10s %10s" % ("情况", "噪声", "SQNR(线性)", "SQNR(dB)"))
    for label, test in [
        ("完全一致", ref.copy()),
        ("每个元素 +0.01", ref + 0.01),
        ("每个元素 +0.1", ref + 0.1),
        ("每个元素 +1.0", ref + 1.0),
        ("每个元素 +10.0", ref + 10.0),
    ]:
        err = test - ref
        noise = float(np.sum(err ** 2))
        ratio = float(np.sum(ref ** 2)) / noise if noise > 0 else float("inf")
        db = 10 * np.log10(ratio) if noise > 0 else float("inf")
        print("  %-30s %-20.4f %10.2f %10.2f" % (label, noise, ratio, db))
    print()
    print("  公式: SQNR = 10 * log10( sum(ref^2) / sum(err^2) )")
    print("    分子 = 参考信号的能量（ref = reference，标准答案）")
    print("    分母 = 误差的能量")
    print("    10*log10 = 转成 dB（分贝），因为线性比值跨度太大不好读")
    print()
    print("  为什么分子用 ref 而不是 test？")
    print("    因为 ref 才是'真正想要的信号'。我们要衡量的是：")
    print("    '相对于我想要的信号，噪声有多大'")
    print("    如果分子用 test，那衡量的是'相对于噪声，我输出的能量有多大'——意思全变了。")
    print("    这也是为什么 sqnr(reference, test) 的参数顺序不能反。")
    print()
    print("  >>> 为什么量化分析用它：")
    print("      cosine 只看方向，幅度错了它不知道。")
    print("      SQNR 同时看方向和幅度，能发现'整体被压缩了 10 倍'这种问题。")
    print("      实测：per-tensor 最差层 SQNR = 5.29 dB，意味着噪声能量是信号的 1/3。")
    print("      经验值：>30 dB 基本无损，<15 dB 这层基本废了。")
    print()

    print("=" * 78)
    print("三、两者为什么都需要（一个具体反例）")
    print("=" * 78)
    ref = np.array([1.0, 2.0, 3.0])
    scaled_wrong = ref * 0.001          # 方向对，但幅度几乎归零
    noisy_wrong = ref + np.array([0.5, -0.5, 0.5])   # 幅度对，但误差大
    print("  参照 ref = %s" % ref.tolist())
    print()
    for label, test in [("幅度被压到 0.1%", scaled_wrong), ("加了较大噪声", noisy_wrong)]:
        print("  %-18s test=%s" % (label, np.round(test, 3).tolist()))
        print("  %-18s cosine=%.6f   SQNR=%.2f dB" % ("", cosine(ref, test), sqnr(ref, test)))
        print()
    print("  >>> 第一种情况 cosine 仍然是 1.0（方向完全一致），但 SQNR 极低。")
    print("      只用 cosine 会漏掉'信号被削弱到几乎消失'这种严重问题。")
    print("      这就是为什么报告里两个指标都要报。")


if __name__ == "__main__":
    main()
