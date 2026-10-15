"""Demonstrate why np.abs is required, and what .ravel() does and does not change.

Teaching material for LAB lesson 5.
"""

import numpy as np


def main() -> None:
    print("=" * 76)
    print("实验 1：去掉 np.abs 会怎样")
    print("=" * 76)
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    test = np.array([1.0, 2.0, 3.0, 4.5], dtype=np.float64)
    err = test - ref
    print("  FP32 值 :", ref)
    print("  INT8 值 :", test)
    print("  误差    :", err)
    print()
    print("  有 np.abs : np.max(np.abs(err)) = %.4f   <- 真实的'最大偏离'"
          % np.max(np.abs(err)))
    print("  无 np.abs : np.max(err)          = %.4f   <- 只是'最大的正偏差'"
          % np.max(err))
    print()

    # 最坏情况：所有误差都是负的
    ref2 = np.array([5.0, 6.0, 7.0])
    test2 = np.array([4.0, 5.0, 6.0])           # 每项都错了 1.0
    err2 = test2 - ref2
    print("  如果所有误差都是负的：")
    print("    FP32 :", ref2)
    print("    INT8 :", test2)
    print("    误差 :", err2, " -> 每个都错了 1.0")
    print("    有 np.abs : %.4f   <- 正确报告了 1.0 的误差" % np.max(np.abs(err2)))
    print("    无 np.abs : %.4f   <- 报告'误差 -1.0'，但更糟的是……" % np.max(err2))
    ref3 = np.array([5.0, 6.0, 7.0])
    test3 = np.array([4.9, 5.9, 6.9])           # 全部偏小 0.1
    err3 = test3 - ref3
    print()
    print("  如果误差全是负数但大小不同：")
    print("    误差 :", err3)
    print("    有 np.abs : %.4f" % np.max(np.abs(err3)))
    print("    无 np.abs : %.4f   <- 取了'最接近 0 的那个负误差'！" % np.max(err3))
    print()
    print("  >>> 结论：np.max 在负数里挑的是'最大的那个'，也就是最接近 0 的。")
    print("      所以没有 np.abs 时，误差越大反而报告得越小 —— 完全反了。")
    print()

    print("=" * 76)
    print("实验 2：.ravel() 会不会改变 np.max(np.abs(err)) 的结果")
    print("=" * 76)
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 12.0
    b = a + np.array([[[0.5, -3.0, 0.1, 0.2]] * 3] * 2)
    err = b - a
    print("  a 的形状:", a.shape)
    print("  err 的形状:", err.shape)
    print("  逐元素相减：numpy 按位置对应相减，和形状是几维无关")
    print()
    v_multi = np.max(np.abs(err))
    v_flat = np.max(np.abs(err).ravel())
    print("  不 ravel : np.max(np.abs(err))         = %.4f" % v_multi)
    print("  ravel 后 : np.max(np.abs(err).ravel()) = %.4f" % v_flat)
    print("  相同吗   : %s" % (v_multi == v_flat))
    print()
    print("  >>> 结论：结果完全一样。ravel 只是改变'排列方式'，不改变元素的值。")
    print()

    print("=" * 76)
    print("那 .ravel() 到底有什么用")
    print("=" * 76)
    print("  1. 一致性：不管输入是 (1,320,28,28) 还是 (1000,)，代码都一样，")
    print("     不用为不同形状写不同分支。")
    print("  2. 代码意图更清楚：想表达'我不关心形状，只想看全部元素'。")
    print()
    print("  什么时候形状才真正重要？")
    x = np.array([[1., 2., 3.]])          # 形状 (1, 3)
    y = np.array([[10.], [20.]])          # 形状 (2, 1)
    print("    x 形状 %s, y 形状 %s" % (x.shape, y.shape))
    print("    x + y =")
    print(x + y)
    print("    -> 这叫'广播(broadcasting)'：两个形状不同的数组做运算时，")
    print("       numpy 会自动扩展它们。这才是形状真正起作用的地方。")
    print("    而 'a - b'（形状相同时）和 'np.max(...)' 都不涉及广播。")


if __name__ == "__main__":
    main()
