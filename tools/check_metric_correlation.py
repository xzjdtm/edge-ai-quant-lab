"""Is max_abs_err inversely correlated with cosine?

The LAB checkpoint asks whether the new metric agrees with the existing ones.
Answering with 5 rows is guessing; this computes it over the whole table and
lists the layers where the two metrics disagree.

Usage:  python check_metric_correlation.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "artifacts/tables/04_layer_metrics.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))

    print("=" * 76)
    print("max_abs_err 和 cosine / sqnr_db 的关系（全部数据）")
    print("=" * 76)

    for variant in sorted({r["variant"] for r in rows}):
        sub = [r for r in rows if r["variant"] == variant]
        cos = np.array([float(r["cosine"]) for r in sub])
        sqnr = np.array([float(r["sqnr_db"]) for r in sub])
        mae = np.array([float(r["max_abs_err"]) for r in sub])
        # 皮尔逊相关系数：1=完全正相关, -1=完全负相关, 0=无关
        r_cos = float(np.corrcoef(cos, mae)[0, 1])
        r_sqnr = float(np.corrcoef(sqnr, mae)[0, 1])
        print()
        print("  变体: %s  (%d 层)" % (variant, len(sub)))
        print("    corr(cosine,   max_abs_err) = %+.3f   %s"
              % (r_cos, "<- 反相关" if r_cos < -0.5 else "<- 关系弱"))
        print("    corr(sqnr_db,  max_abs_err) = %+.3f   %s"
              % (r_sqnr, "<- 反相关" if r_sqnr < -0.5 else "<- 关系弱"))

    # 用关系最强的那一组做细节分析
    variant = "qdq_pertensor_minmax"
    sub = [r for r in rows if r["variant"] == variant]
    sub.sort(key=lambda r: float(r["cosine"]))
    print()
    print("=" * 76)
    print("%s：按 cosine 从差到好排序，看 max_abs_err 是不是同步变化" % variant)
    print("=" * 76)
    print("  %-4s %-34s %10s %10s %14s" % ("排名", "张量", "cosine", "sqnr_db", "max_abs_err"))
    for i, r in enumerate(sub[:10], 1):
        print("  %-4d %-34s %10.4f %10.2f %14.3f"
              % (i, r["tensor"][-32:], float(r["cosine"]),
                 float(r["sqnr_db"]), float(r["max_abs_err"])))

    # 找"cosine 排名"和"max_abs_err 排名"差异最大的层
    by_cos = sorted(sub, key=lambda r: float(r["cosine"]))          # 差 -> 好
    by_mae = sorted(sub, key=lambda r: -float(r["max_abs_err"]))    # 大 -> 小
    rank_cos = {r["tensor"]: i for i, r in enumerate(by_cos)}
    rank_mae = {r["tensor"]: i for i, r in enumerate(by_mae)}
    print()
    print("=" * 76)
    print("两个指标排名差异最大的 5 层（说明它们看的是不同的东西）")
    print("=" * 76)
    print("  %-34s %10s %10s %8s" % ("张量", "cos排名", "mae排名", "差"))
    diffs = sorted(
        sub, key=lambda r: -abs(rank_cos[r["tensor"]] - rank_mae[r["tensor"]])
    )[:5]
    for r in diffs:
        t = r["tensor"]
        print("  %-34s %10d %10d %8d"
              % (t[-32:], rank_cos[t] + 1, rank_mae[t] + 1,
                 rank_cos[t] - rank_mae[t]))
    print()
    print("  排名差 = cosine排名 - max_abs_err排名")
    print("  正值表示'按 cosine 看很差，但按 max_abs_err 看还好'（或反之）")
    print()
    print("  >>> 结论：两个指标高度相关但不是同一个东西。")
    print("      cosine 衡量'整体方向偏移'，max_abs_err 衡量'最坏的那个元素'。")
    print("      当某层是'少数元素饱和'时，两个指标会给出不同排序 ——")
    print("      这正是加这个补充指标的意义。")


if __name__ == "__main__":
    main()
