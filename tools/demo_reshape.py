"""Visualize what reshape does, and why broadcasting sometimes works silently.

Teaching material for LAB lesson 5.
"""

import numpy as np


def show(label, arr):
    print("  %-22s shape=%-14s 值=%s" % (label, str(arr.shape), arr.ravel().tolist()))


def main() -> None:
    print("=" * 74)
    print("reshape 做了什么：不复制数据，只换一个'看法'")
    print("=" * 74)
    a = np.arange(6)
    show("原始 a", a)
    b = a.reshape(2, 3)
    show("a.reshape(2,3)", b)
    c = a.reshape(1, 2, 3)
    show("a.reshape(1,2,3)", c)
    d = a.reshape(6, 1)
    show("a.reshape(6,1)", d)
    print()
    print("  注意：所有形状下，.ravel() 展开的顺序完全一样 -> 没有复制、没有重排")
    print("  a.reshape(2,3) 可以把内存画成:")
    print("     [[0 1 2]")
    print("      [3 4 5]]")
    print()

    print("=" * 74)
    print("关键证明：reshape 不复制内存（改的是 strides，不是 data）")
    print("=" * 74)
    x = np.arange(6)
    y = x.reshape(2, 3)
    print("  x 的 data 内存地址 : %d" % x.__array_interface__["data"][0])
    print("  y 的 data 内存地址 : %d" % y.__array_interface__["data"][0])
    print("  是同一块内存吗     : %s" % (x.__array_interface__["data"][0] == y.__array_interface__["data"][0]))
    print()
    print("  验证：改 y 的元素，x 跟着变（说明确实是同一块内存）")
    y[0, 0] = 999
    print("    y[0,0] = 999 之后, x =", x.tolist())
    print()

    print("=" * 74)
    print("那什么时候 reshape 会真的复制？（不是总能当'视图'）")
    print("=" * 74)
    z = np.arange(12).reshape(3, 4)
    show("原始 z", z)
    t = z.T                      # 转置：只是换 strides，不复制
    print("  z.T 是视图吗: %s" % (t.base is z))
    try:
        r = t.reshape(12)        # 对非连续内存 reshape 会复制
        print("  z.T.reshape(12) 成功, 值 =", r.tolist())
        print("  但它是副本: %s" % (r.base is None))
    except Exception as exc:
        print("  失败:", exc)
    print()

    print("=" * 74)
    print("回到你的问题：a 是 (1,1000)，b 是 (1,1,1000)")
    print("=" * 74)
    a = np.arange(1, 5).reshape(1, 4)
    b = np.arange(1, 5).reshape(1, 1, 4)
    show("a", a)
    show("b", b)
    print()
    print("  情况 1：b.reshape(a.shape)  ->  按 a 的形状重塑")
    br = b.reshape(a.shape)
    show("b.reshape((1,4))", br)
    print("  现在形状一致，直接相减:")
    show("a - br", a - br)
    print()
    print("  情况 2：不 reshape，直接相减（靠广播）")
    show("a - b (广播)", a - b)
    print()
    print("  >>> 两种方式结果一样！因为 b 的 (1,1,4) 里那个多余的 1 不改变任何东西。")
    print("      广播从最右维度对齐，前面自动补 1，所以 (1,4) 和 (1,1,4) 是兼容的。")
    print()
    print("  真正需要 reshape 的是这种情况：")
    p = np.arange(4).reshape(1, 4)
    q = np.arange(4).reshape(4, 1)
    show("p", p)
    show("q", q)
    print("  p - q (4,4) 会广播成一个大矩阵，这几乎肯定是错的:")
    print(p - q)
    print()
    print("  >>> 这时 a.shape != b.shape 且元素总数相同，reshape 才有意义：")
    print("      q.reshape(1,4) 之后相减 = ", (p - q.reshape(1, 4)).tolist())


if __name__ == "__main__":
    main()
