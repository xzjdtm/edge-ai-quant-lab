"""Show why numpy exists: vectorized ops vs explicit Python loops.

Teaching material for LAB lesson 5 (numpy introduction).
"""

import time

import numpy as np


def main() -> None:
    n = 2_000_000                      # 200 万个数，接近一个小特征图的规模
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)

    print("=" * 74)
    print("同一个任务：把两个 200 万长度的数组逐元素相加")
    print("=" * 74)
    print()

    # ---- 方式 1：纯 Python 循环 ----
    print("[方式 1] 纯 Python 循环（就相当于 C 里的 for 循环）")
    lst_a, lst_b = a.tolist(), b.tolist()
    t0 = time.perf_counter()
    result_loop = [x + y for x, y in zip(lst_a, lst_b)]
    t_loop = time.perf_counter() - t0
    print("  代码:  result = [x + y for x, y in zip(a, b)]")
    print("  耗时:  %.3f 秒" % t_loop)
    print()

    # ---- 方式 2：numpy 向量化 ----
    print("[方式 2] numpy 向量化")
    t0 = time.perf_counter()
    result_np = a + b
    t_np = time.perf_counter() - t0
    print("  代码:  result = a + b")
    print("  耗时:  %.6f 秒" % t_np)
    print()

    print("=" * 74)
    print("结果")
    print("=" * 74)
    print("  快了 %.0f 倍" % (t_loop / max(t_np, 1e-9)))
    print("  结果一致: %s" % np.allclose(np.array(result_loop), result_np))
    print()
    print("  为什么快这么多：")
    print("    Python 循环 : 每次迭代都要做类型检查、创建 Python 对象，200 万次")
    print("    numpy       : 底层是 C 循环，且能利用 CPU 的 SIMD 指令一次算多个数")
    print()
    print("=" * 74)
    print("这就是为什么深度学习全用 numpy（以及建立在它之上的 PyTorch）")
    print("=" * 74)
    print("  神经网络的本质就是海量数组运算：")
    print("    一个 Conv 层的输出 = 权重数组 和 输入数组 的一堆乘加")
    print("    用 Python 循环写，训练一个模型要跑几个月")
    print("    用 numpy/PyTorch，几小时")
    print()
    print("numpy 里你在这个项目会用到的几个操作：")
    print("  np.asarray(x)      把别的类型转成 ndarray")
    print("  x.ravel()          把多维数组拍平成一维")
    print("  x.shape            形状，比如 (1, 320, 28, 28)")
    print("  x.dtype            数据类型，比如 float32 / int8")
    print("  x.nbytes           占多少字节 = 元素个数 x 每个元素的字节数")
    print("  np.max / np.abs    取最大值 / 取绝对值（对整个数组一次算完）")
    print("  a - b              逐元素相减")
    print("  np.dot(a, b)       点积（cosine 相似度里用到）")
    print("  np.linalg.norm(x)  向量长度")


if __name__ == "__main__":
    main()
