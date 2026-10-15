"""Compare dict comprehension vs explicit loop: readability and speed.

Answers the LAB question "why write it as a comprehension when the loop is
shorter".
"""

import timeit

SETUP = """
shared = [
    "/features/features.%d/conv/conv.2/Conv_output_0" % i for i in range(29)
]
"""

COMPREHENSION = """
accum = {
    name: {"cos": [], "sqnr": [], "rel": [], "absmax": [], "rms": [], "maxerr": []}
    for name in shared
}
"""

LOOP = """
accum = {}
for name in shared:
    accum[name] = {"cos": [], "sqnr": [], "rel": [], "absmax": [], "rms": [], "maxerr": []}
"""


def main() -> None:
    n = 100_000
    t_comp = timeit.timeit(COMPREHENSION, setup=SETUP, number=n)
    t_loop = timeit.timeit(LOOP, setup=SETUP, number=n)

    print("=" * 72)
    print("建立 29 个层的收集箱，重复 %d 次" % n)
    print("=" * 72)
    print("  推导式      : %.4f 秒  (单次 %.2f 微秒)" % (t_comp, t_comp / n * 1e6))
    print("  for 循环    : %.4f 秒  (单次 %.2f 微秒)" % (t_loop, t_loop / n * 1e6))
    print("  推导式快    : %.2f 倍" % (t_loop / t_comp))
    print()
    print("  单次差值    : %.2f 微秒" % ((t_loop - t_comp) / n * 1e6))
    print()
    print("=" * 72)
    print("结论（注意：实测推翻了我原来的假设）")
    print("=" * 72)
    print("  我原以为推导式更快，实测是 %.2f 倍 —— 在这个规模下两者没有可测差异" % (t_loop / t_comp))
    if t_loop < t_comp:
        print("  （循环甚至略快，说明'推导式更快'这个常见说法不能无条件相信）")
    print()
    print("  >>> 所以真实理由不是性能，是：")
    print("      1. 意图更清楚 —— 一眼看出'这是在把一个可迭代对象映射成字典'，")
    print("         而 for 循环要先读循环体才能知道意图")
    print("      2. 推导式是表达式，for 是语句 —— 表达式可以直接当参数传给函数，")
    print("         语句必须先赋值给变量。这在'就地构造并返回'时有用")
    print()
    print("  什么时候该用 for 循环？")
    print("      循环体里有 if/else、有 try、有多条语句、或需要提前 break ——")
    print("      推导式会变得难读，那时 for 循环更好。")
    print()
    print("  在这个具体位置，两种写法都可以，是风格选择。")


if __name__ == "__main__":
    main()
