# W1 完整解析：这条流水线到底做了什么

> 这份文档的目的不是总结成果，而是让你能**自己讲清楚每一步在干什么、为什么这么干**。
> 读的方式：每节先看「这段在解决什么问题」，再看「代码怎么做」，然后**打开对应脚本对着看**。
> 有不懂的地方直接问我，我会拆得更细。

---

## 0. 全局：一条量化分析流水线长什么样

```
        PyTorch 模型 (.pt)
              │  ① 导出
              ▼
        ONNX 图 (.onnx)  ──────► ② ONNX Runtime 推理（先确认它能跑对）
              │  ③ 量化：插桩分析图 + 校准
              ▼
        INT8 模型 (.onnx, QDQ 格式)
              │  ④ 对比：把两张图的中间激活都暴露出来，逐层算相似度
              │  ⑤ 评估：端到端精度 / 分布漂移
              ▼
        结论：哪一层崩了、per-channel 值多少、校准集要多少张
```

每一段回答一个具体问题：

| 段 | 回答的问题 | 脚本 |
|---|---|---|
| ① 导出 | ONNX 里到底有什么？导出踩什么坑？ | `01_export_onnx.py` |
| ② 推理 | 怎么跑？provider/batch/优化级别影响多大？ | `08_ort_inference.py` |
| ③ 量化 | INT8 是怎么算出来的？校准集怎么用？ | `03_quantize.py` |
| ④ 逐层对比 | **是哪一层开始崩的？** | `04_layer_analysis.py` |
| ⑤ 评估 | 崩了以后任务精度掉多少？ | `05_accuracy.py` / `06_calib_sweep.py` |

**④ 是整件事的核心。** 因为"量化掉点"这个现象人人都知道，能说出"误差随层深累积，层序号与 cosine 的相关系数是 -0.63"才是能力。

---

## 一、ONNX 导出（`01_export_onnx.py`）

### 1.1 为什么要有 ONNX 这一层

PyTorch 模型是一个 Python 对象，里面是 `nn.Module` 的嵌套结构。这个结构**只有 PyTorch 自己认识**。板子上的推理引擎（RKNN、nncase、ncnn）不是 PyTorch，它们需要一个**语言无关的、只描述计算的图**。

ONNX 就是那个中间语言：一张**有向无环图（DAG）**，节点是算子（Conv/Relu/Add），边是张量。

类比：PyTorch 模型像一份 Word 文档，ONNX 像一份 PDF。丢了一些可编辑性，但谁都能读。

### 1.2 代码做了什么

```python
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,          # ①
    dynamic_axes=None,         # ②
    do_constant_folding=True,  # ③
)
```

**① `opset_version` 是什么，为什么选 13**

ONNX 的算子集是有版本的。每一版可能新增算子、也可能改算子的定义。这里有个坑：

- 版本太低 → 某些算子不存在，导出报错
- 版本太高 → 板子的推理引擎还没支持

13 是个常见的保守选择。**关键认知：这决定了图里能出现哪些算子，直接影响下游工具链能不能吃。**

**② `dynamic_axes` 是什么**

导出的图默认形状是**写死的**：输入 `[1, 3, 224, 224]`。这个 `1` 就是 batch。

- 不设 `dynamic_axes` → 图里 batch 固定为 1，你喂 4 张图会报错
- 设了 → batch 变成符号 `"batch"`，可以喂任意数量

**这是一把双刃剑**：动态 shape 灵活，但推理引擎没法提前分配内存、也没法做针对固定形状的优化。**端侧部署通常反而要固定 shape**，因为固定才能把内存和延迟压到最低。

我们这次的图就是固定 batch=1，所以后面所有脚本都要处理"我想一次喂 4 张但图不答应"的问题（见 3.3）。

**③ `do_constant_folding` 是什么**

有些算子的输入全是常量，比如 `x = 1 + 2`。折叠就是**在导出时直接算成 `x = 3`**，把节点消掉。省的是**推理时的算子调用**，不是显存。

### 1.3 onnxsim 做了什么——这个数字值得记

导出得到 170 个节点，`onnxsim` 简化后 100 个。我让脚本去查了**具体删了什么**：

```
op_type                before  after   delta
Constant                   70      0     -70
```

**删掉的全部是 70 个 `Constant` 节点**，它们被折叠进了 initializer。

**但文件大小一点没变（都是 13.99 MB）。** 因为常量折叠改变的是**图的形状**（节点少了、权重多了），不是模型体积。权重字节一个没少，只是从"算子节点"变成了"initializer"。

> 面试点：说"onnxsim 把模型压小了"是错的。它让**推理时少执行 70 个算子**，文件大小不变。

### 1.4 怎么验证简化没改坏

**这一步不能省。** 简化是在改图，改错了你的所有结论都是垃圾。

```python
sess = ort.InferenceSession(sim_path)
got = sess.run(None, {"input": dummy.numpy()})[0]
diff = np.max(np.abs(got - ref))          # ref 是 PyTorch 的输出
# 实测：max|diff| = 8.106e-06
```

`8e-06` 是浮点累计误差量级，说明**简化没有改变计算语义**。

> 面试点：量化/简化前后必须做数值验证，并且要给出量化指标（max abs diff / cosine），不能只说"看起来一样"。

---

## 二、ONNX Runtime 推理（`08_ort_inference.py`）

### 2.1 provider 的真实含义

```python
ort.get_available_providers()   # 本机: ['CPUExecutionProvider']
ort.get_device()                # 'CPU'
```

`providers` 是一个**按优先级排列的列表**。ONNX Runtime 会用第一个能成功初始化的。

**关键陷阱**：在没有 GPU 的机器上写 `providers=["CUDAExecutionProvider"]` **不会报错**，它会静默回退到 CPU。所以：

```python
sess.get_providers()   # 必须查这个，才知道真正生效的是谁
```

> 面试点：这是"你以为在用 GPU 其实在用 CPU"的经典事故来源。

### 2.2 `SessionOptions` 里真正影响性能的三个东西

```python
opts = ort.SessionOptions()
opts.graph_optimization_level = ...   # ①
opts.intra_op_num_threads = 4         # ②
opts.inter_op_num_threads = ...       # ③
```

**① 图优化级别**——实测影响最大：

| 级别 | 做了什么 | 实测 median |
|---|---|---|
| `ORT_DISABLE_ALL` | 不优化 | 26.99 ms |
| `ORT_ENABLE_BASIC` | 常量折叠、冗余节点消除 | 26.95 ms |
| `ORT_ENABLE_EXTENDED` | + 算子融合（Conv+BN+Relu → 一个） | 24.86 ms |
| `ORT_ENABLE_ALL` | + 针对硬件的布局优化 | **12.37 ms** |

**2.18 倍**，全部来自"算子融合 + 内存布局"。这是本机最大的单点收益，比调线程数、比 warmup 都值钱。

`intra` = 单个算子内部用几个线程（矩阵乘可以并行）
`inter` = 不同算子之间并行（有依赖关系，所以收益有限）

实测 intra：1/2/4 线程 → 12.38 / 7.19 / 3.91 ms。**注意这不是线性的**（4 线程理论该 3.1ms，实际 3.91ms），因为有线程调度开销和内存带宽瓶颈。

### 2.3 warmup：一个被讲烂但实测很反直觉的点

```python
sess.run(...)              # 第一次：首帧
first = time.perf_counter() - t0

for _ in range(3):         # warmup
    sess.run(...)

times = [ ... ]            # 才开始正式计时
```

**为什么必须 warmup**：第一次调用要分配内存 arena、选择/编译 kernel。

**但实测收益只有 1.02 倍**（12.66ms → 12.37ms）。为什么和常说的"好几倍"不符？

因为**收益取决于后端**：

| 场景 | warmup 收益 | 原因 |
|---|---|---|
| CPU + 小模型（我们） | ~1.02x | 没有 kernel 调优，arena 也不大 |
| **CUDA / TensorRT** | **10x+** | cuDNN 要 autotune 卷积算法，可能试几十种 |
| 大模型 | 高 | arena 分配占比大 |
| 做 p95/p99 统计 | 必须要 | 第一次的异常值会污染分位数 |

> 面试点：正确说法是"**必须做 warmup，但收益大小取决于后端**"。直接说"不做 warmup 会慢好几倍"在你的 CPU 场景下是错的，被追问就拿不出数字。

### 2.4 IO binding 是什么

普通 `run()` 每次调用：
```
numpy 数组 ──拷贝──► ORT 输入张量 ──计算──► ORT 输出张量 ──拷贝──► numpy 数组
```

IOBinding 让你自己指定内存，省掉这两次拷贝。在 CPU 上省得少（内存是同一块），但在 CUDA 上省的是**每帧的 H2D/D2H 传输**，那是实打实的延迟。

```python
io = sess.io_binding()
x_ort = ort.OrtValue.ortvalue_from_numpy(x, "cpu", 0)
io.bind_ortvalue_input(name, x_ort)      # 注意这个 API 名，容易写错
io.bind_output(out_name, "cpu")
sess.run_with_iobinding(io)
```

---

## 三、量化：原理和实现（`03_quantize.py`）

### 3.1 量化到底在算什么

一个 FP32 张量 `x`，要变成 INT8 的整数 `q`：

```
q = round(x / scale) + zero_point        # 量化
x' = (q - zero_point) * scale            # 反量化
```

- **scale**：一个浮点数，表示"1 个整数刻度代表多大的实数"
- **zero_point**：一个整数，表示"实数 0 对应哪个整数"

为什么需要 zero_point？因为 ReLU 之后的激活全是非负的。如果强行用对称量化（zero_point=0），那 `[0, 6]` 这个范围会映射到 `[-127, 127]`，**一半的整数刻度浪费在负数区**。用非对称量化（zero_point=某个负数）就能把 256 个刻度全用在 `[0, 6]` 上。

**对称 vs 非对称**

| | 对称 | 非对称 |
|---|---|---|
| zero_point | 恒为 0 | 可调 |
| 范围 | `[-max_abs, max_abs]` | `[min, max]` |
| 适合 | 权重（近似对称分布） | 激活（ReLU 后非负） |
| 计算 | 快（没有 zp 加减） | 慢一点 |

实测数据里能看到：`input` 这层的 `zp=-14`（非对称），`Clip_output_0` 的 `zp=-128`（这层被量到 INT8 的最小值，说明它的范围几乎是纯正的）。

### 3.2 误差从哪来

两个来源：

1. **权重舍入**：`round()` 本身有误差，最大 `scale/2`
2. **激活截断**：校准集里没出现过的值，推理时超出了 `[min, max]` 范围，被**夹断（saturate）**

第 2 点是校准集存在的全部理由。**校准集的任务就是"估计每一层激活的真实动态范围"。**

### 3.3 代码怎么做的

```python
quantize_static(
    src, dst, calibration_data_reader,
    quant_format=QuantFormat.QDQ,          # ①
    per_channel=True,                      # ②
    calibrate_method=CalibrationMethod.MinMax,  # ③
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)
```

**① `QDQ` vs `QOperator` 两种格式**

| 格式 | 做法 | 图里出现什么 | 实测节点数 |
|---|---|---|---|
| `QDQ` | 在算子输入输出处插 `QuantizeLinear`/`DequantizeLinear` 对，**保留原始 FP32 算子** | 66 个 Q + 172 个 DQ | |
| `QOperator` | 把算子整个替换成量化版本 | 2 个 Q + 2 个 DQ + **52 个 QLinearConv** | |

**为什么用 QDQ 做分析**：因为它保留了原始的 Conv/Add 节点和张量名，两张图才可能逐层对齐。QOperator 把算子换掉了，名字全变了。

实际部署时选哪个看工具链：RKNN/nncase 通常内部会做类似 QOperator 的融合。

**② `per_channel` 是最关键的一个参数**

- `per_channel=False`：整个权重张量共用一个 scale
- `per_channel=True`：**每个输出通道一个 scale**

为什么重要？想象一个卷积层有 64 个输出通道，其中 63 个的值都在 `[-0.1, 0.1]`，只有 1 个在 `[-10, 10]`。

- per-tensor：scale 由那个最大的通道决定，`scale ≈ 10/127 ≈ 0.079`。其他 63 个通道的值只有 ±0.1，除以 0.079 之后只剩 ±1.3 → **量化后基本上全是 0 或 ±1，精度全丢**
- per-channel：正常通道用自己的 scale（`0.1/127 ≈ 0.0008`），能精细表示

**实测差距**：top-1 一致率 0.84 → 0.98，KL 散度降 21 倍。代价只有 +0.24 MB。

**注意**：`per_channel` 只管**权重**，激活永远是 per-tensor（因为激活范围依赖输入数据，没法固定按通道给 scale）。

**③ `calibrate_method` 校准算法**

| 方法 | 思路 |
|---|---|
| `MinMax` | 直接取观测到的 `[min, max]`。对离群值敏感 |
| `Entropy` | 最小化量化前后分布的 KL 散度，自动找一个"丢弃多少尾部"的阈值 |
| `Percentile` | 直接砍掉 0.001% 的尾部，抗离群值 |

实测这三个在本模型上几乎没差别（`Entropy` 和 `MinMax` 结果完全一样，`Percentile` 的 KL 略好 0.0078 vs 0.0091）。

### 3.4 校准集是怎么被消耗的

```python
class ImageCalibrationReader:
    def get_next(self):
        # 每次返回 {输入名: numpy 数组}，直到返回 None
```

ONNX Runtime 会循环调 `get_next()`，把每批数据喂进图，**在前向过程中记录每一层激活的 min/max**，跑完再根据这些统计量算 scale。

**所以校准集的作用是"探测数据分布"，不是"训练"。** 它不需要标签。

### 3.5 一个必须处理的工程细节：batch 冲突

导出的图 batch 固定为 1。我一开始按 4 张一批喂，直接报错：

```
INVALID_ARGUMENT: Got invalid dimensions for input: input
 index: 0 Got: 4 Expected: 1
```

修法是**从图里读声明**，而不是假设：

```python
def effective_batch(session, requested):
    declared = session.get_inputs()[0].shape[0]
    if isinstance(declared, int) and declared > 0:
        return declared
    return requested
```

> 面试点：这是"固定 shape 部署"的真实代价——所有上下限、批处理逻辑都要跟着改。

---

## 四、逐层误差分析（`04_layer_analysis.py`）——整件事的核心

### 4.1 思路

要找出"哪一层开始崩"，就得拿到**同一层在 FP32 和 INT8 下的输出**，然后比。

问题是：ONNX 图默认**只输出最终结果**。中间层算了就丢了。

解决办法：**改图，把中间张量提升成图的输出**。

```python
for node in model.graph.node:
    if node.op_type in ("Conv", "Add", ...):
        for out in node.output:
            model.graph.output.append(value_info(out))   # 提升为输出
onnx.save(model, out_path)
```

这样跑一次推理，就能一次性拿到所有层的激活。

### 4.2 一个坑（也是本工作里最值得讲的一段）

**第一版做法**：遍历所有 `Conv` 的输出，把名字记下来，在 FP32 图和 INT8 图里都暴露这些名字，然后比。

**结果**：只有 **17/53** 个卷积层能对齐。

#### 我当时的（错误）判断

我怀疑是 ONNX Runtime 的 QDQ 量化器**吞掉了 `Clip` 节点并把输出名复用给了前面的 Conv**，
于是推断那 17 个同名张量在两张图里指向**不同的计算**（FP32 是激活前、INT8 是激活后），
数值毫无意义。

这个解释当时被我写进了报告。**它是错的。**

#### 怎么证伪的

做了个对照实验：**用两种选层方式跑同一份数据，比较重叠张量的数值。**

- 方式 A（对照）：只取 `Conv` 输出名 → 18 层
- 方式 B（正确）：全算子候选 + 张量名交集 → 29 层

如果两种方式选到的同名张量**语义不同**，那同一个张量名在 A 和 B 下应该给出**不同的 cosine**。

实测结果：

| | 方式 A | 方式 B |
|---|---|---|
| 重叠张量的最差 cosine | 0.980080 | 0.980080 |
| 重叠张量的最好 cosine | 0.998286 | 0.998286 |

**完全相同 ⟹ 语义相同 ⟹ 我的假设被证伪。**

#### 真实的问题

不是"比错了"，而是**"看得不全"**：

| 方式 | 层数 | 漏掉了什么 |
|---|---|---|
| Conv-only | 17（+图输出 = 18） | 10 个 `Add`（残差）、`GlobalAveragePool`、`Gemm`（分类头） |
| 全算子交集 | **29** | 无 |

**为什么漏掉 Add 很严重**：看报告第 3 节，per-channel 的最差层列表里
`/features/features.13/Add_output_0` 就在里面。**只取 Conv 就永远看不到残差连接的表现**，
而残差连接要处理两路数值范围不同的张量，是量化时需要重点关注的结构。

> ⚠️ **注意别过度解读**：我后来用全量数据核对过，**Add 和普通 Conv 的平均 cosine 几乎相同**
> （per-channel 0.9880 vs 0.9879；per-tensor 0.9178 vs 0.9071）。
> Add 出现在最差榜前列是因为它**数量多（10/29）且位置靠后**，不是因为算子本身更脆弱。
> **准确的说法是"漏掉它会丢掉三分之一的观测层"，而不是"它最脆弱"。**

#### 教训

我的第一反应是**往深处想**（"一定是算子融合导致语义错位"），真相却在**浅处**（"我的筛选条件太窄"）。

> **越是听起来专业的解释，越容易让你停止验证。**

面试时的正确讲法：
> "我一开始只取 Conv 输出，只有 17 层能对齐。我先假设是量化器算子融合导致语义错位，
> 但做了对照实验后发现两种方式下重叠张量的 cosine **完全相同**，说明语义一致，
> 我的假设被证伪了。真实问题是我漏掉了 10 个残差 Add 和池化头——
> 好在后来核对数据发现 Add 本身并不比 Conv 更脆弱，所以这个 bug 的实质是
> **丢了三分之一的观测层**，不是"漏掉了最脆弱的层"。改成全算子交集后拿到 29 层。"

这段话证明的是**你会用实验推翻自己的假设**，比"我会写代码"值钱得多。

动手复现：`LAB.md` 第 4.3 节。

### 4.3 顺便还解决了"执行顺序"问题

节点的**存储顺序**不等于**执行顺序**。ONNX 里节点是按拓扑排序存的，但同一个"深度"的节点顺序是任意的。

所以 `candidate_tensors()` 做了一次从输入出发的**深度优先遍历**，按真实数据流顺序给每层编号。这样画出来的曲线横轴才有意义。

### 4.4 三个指标，为什么需要三个

| 指标 | 公式 | 看什么 |
|---|---|---|
| **cosine** | `a·b / (‖a‖‖b‖)` | 方向是否保持。1.0 = 完全同向。**对整体缩放不敏感** |
| **SQNR** | `10·log10(Σa²/Σ(a-b)²)` | 信噪比，dB。对幅度变化敏感 |
| **rel_l2** | `‖a-b‖/‖a‖` | 相对误差 |

为什么 cosine 不够？因为**如果一层输出的所有值都被乘了 100 倍但方向不变，cosine 还是 1.0**——但下游的 ReLU、BN 全会受影响。所以要配 SQNR。

实测：per-tensor 最差层 `cosine=0.8387, SQNR=5.29 dB`。5 dB 意味着**噪声能量是信号能量的 1/3**，这层基本废了。

### 4.5 内存管理：为什么要分块跑

把 63 层全暴露成输出，一次推理就要同时持有 63 个中间张量。MobileNetV2 最浅的层是 `1×3×224×224`（600 KB），深的层小一些，但加起来几百 MB。

所以 `compare_models()` 把张量**分块（chunk=24）**跑：每次只让 24 个张量同时存活。

---

## 五、精度评估与校准集实验（`05` / `06`）

### 5.1 为什么不能只看 top-1 一致率

`top1_match` = INT8 和 FP32 选出同一个类的比例。

**这个指标会漏掉一个重要现象**：argmax 没变，但概率分布变了。

举例：FP32 给出 `[0.92, 0.05, 0.03]`，INT8 给出 `[0.51, 0.45, 0.04]`。top-1 一致（都是第 0 类），但第二个类的概率从 0.05 涨到 0.45——**如果下游有任何基于阈值的逻辑（拒识、NMS、告警），行为会完全改变。**

所以还要看：

| 指标 | 公式 | 含义 |
|---|---|---|
| `confidence_drop` | 平均 `p_fp32[argmax] - p_int8[argmax]` | 赢家丢了多少概率 |
| `mean_kl_divergence` | `Σ p_f·log(p_f/p_q)` | **整个分布**的偏移 |
| `top5_contains_fp32_top1` | FP32 的 top1 是否还在 INT8 的 top5 里 | 排序有没有乱 |

实测：per-tensor 的 `confidence_drop` 是 0.1272，per-channel 只有 0.0116。

### 5.2 校准集实验的设计

两个自变量，**必须分开控制**：

**实验 A：规模** —— 10 / 50 / 200 / 500 张，全部均衡采样
**实验 B：多样性** —— 都是 500 张，但构成不同

`data/calib/` 里几种构成：

| 文件 | 构成 | 用意 |
|---|---|---|
| `calib_nested_*.txt` | 从全类别均衡抽 N 张 | 实验 A |
| `calib_narrow_500.txt` | 全部来自 `domain_animal` 域 | 故意"窄"的对照组 |
| `calib_mixed_500.txt` | 一半均衡 + 一半窄域 | 中间的 |
| `calib_domain_balanced_500.txt` | 三个域各抽等量 | 理论上最"均衡"的 |

**为什么"窄"用域而不是"同一类"**：ImageNet 每类只有一张图，凑不出 500 张同类。所以退一步用**域**（按 WordNet class id 排序，前 60% 是动物、60-83% 是物体、后 17% 是场景）。

### 5.3 结果和我的预期不一样，我是怎么处理的

预期（教科书结论）：校准集越具代表性越好 → `domain_balanced_500` 应该最好。

实测：`domain_balanced_500` 反而**最低**（top-1 一致率 0.8438 vs narrow 的 0.9375）。

**我没有把这条抹掉，而是去找原因并写进报告**：

1. 评估集是按 class id 排序后取尾部的 64 张，**物种类占比偏高**，而 `domain_animal` 那 500 张恰好覆盖了剩下的分布
2. 64 张的评估集，4 个百分点的差距**就在采样噪声量级**（64 张里差 2.7 张）

然后给出验证方向：**应该在 per-tensor 配置下重跑这一节**——因为 per-channel 已经让 cosine 稳在 0.99，校准集构成自然退居次要。这正是 W2 混合量化要做的事。

> 面试点：讲"我的预期落空了，原因可能是 A/B，验证方法是 C"比讲一个编圆的故事可信得多。面试官最怕的是"结果完美但讲不出所以然"的人。

---

## 六、踩过的 7 个坑（按耗时排序）

### 坑 1：pip 装不进去（约 40 分钟）★ 最有学习价值

**现象**：
```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied:
  '...\Temp\dsh-xxxx\pip-unpack-xxxx\onnxruntime-1.17.3-cp310-cp310-win_amd64.whl'
```

**排查过程**（这是重点，不是结论）：

| 观察 | 排除的可能 |
|---|---|
| 下载只花 15 秒就成功，只在解压时失败 | ❌ 网络问题 |
| 系统临时目录和工作区临时目录都失败 | ❌ 目录位置问题 |
| PowerShell 自己 `Remove-Item` 也报"访问被拒绝" | ❌ 文件被占用 |
| 100 MB 大文件在工作区写入/读回正常 | ❌ 大文件问题 |
| 对比测试：`os.makedirs(d)` 可写，`tempfile.mkdtemp(d)` 不可写 | ✅ **找到差异** |

**根因**：`tempfile.mkdtemp` 创建目录用的 mode 是 **0700**，而沙箱会拒绝写入 0700 的目录。对照组证明：默认权限 `0777 & ~umask` 的目录写入正常。

而且 `os.mkdir(d, 0o700)` 之后 `os.chmod(d, 0o777)` **也被拒**——创建瞬间就被锁定了，补救无效。

**解法**：在创建时就用默认权限，而不是创建后再改。

```python
def _mkdtemp(suffix="", prefix="tmp", dir=None):
    name = os.path.join(target, prefix + next(tempfile._get_candidate_names()) + suffix)
    os.makedirs(name)      # 默认权限，可写
    return name
```

再加上替换 `TemporaryDirectory`（让清理失败不抛异常），以及每次启动清扫上一轮的残留。

**副产品**：pip 也被绕开了。`tools/fetch_wheel.py` 直接拿 wheel URL → 下载 → `zipfile` 解压 → `PYTHONPATH` 引用。**不改动任何系统环境。**

> 学习点：**"隔离变量"的排查方法**。每次只改一个条件，用最小实验证明差异。不要靠猜。

### 坑 2：Python 3.14 没有可用的 onnx（约 15 分钟）

`base` 环境是 Python 3.14，`onnx` 没有对应 wheel。pip 自动去编译源码，失败。

退一步装了 `onnx 1.16.2` 的 `cp312` wheel，导入时报：
```
ImportError: cannot import name 'ONNX_ML' from 'onnx.onnx_cpp2py_export'
```
**C 扩展的 ABI 不兼容**——cp312 编译的二进制在 3.14 下加载不了。

**解法**：换用 py3.10 的 `rknn-yolov8-train` 环境（那里 `onnx 1.14.1`、torch、onnxsim 都齐）。

> 学习点：Python 版本和 C 扩展 ABI 的绑定关系。`cp310`/`cp312` 这些 tag 不是装饰。

### 坑 3：数据集源全被限速（约 30 分钟）

实测速度：

| 源 | 速度 | 可用 |
|---|---|---|
| cs.toronto.edu（CIFAR-10 官方） | **0.018 MB/s** | ❌ 170MB 要 2.6 小时 |
| robots.ox.ac.uk（Oxford Pets） | **0.03 MB/s** | ❌ 792MB 要 8 小时 |
| mirror.sjtu.edu.cn `cifar/` | 给 5MB 就断 | ❌ 假镜像 |
| hf-mirror.com | 1.2 MB/s | ✅ 但 parquet 解码要 pyarrow，也卡 |
| **raw.githubusercontent.com** | 0.33 MB/s | ✅ 小文件，断了只损失一张 |
| 清华 PyPI 镜像 | **1.6 MB/s** | ✅ |

**解法**：从 GitHub 上的 `EliSchwartz/imagenet-sample-images` **逐张下载**（每张 ~100KB，997 类各一张）。

**这个选择的关键理由**：小文件 + 可重试。断线只损失 1 张，而不是整个归档从头再来。而且 997 类天然提供了丰富的多样性——正好是校准集实验需要的。

> 学习点：**先测速再决定方案，不要假设"官方源就是最好的"**。

### 坑 4：`quant_pre_process` 反复失败（约 30 分钟）

它内部用 `TemporaryDirectory` 存 `optimized.onnx`，然后**再读回来**——正好踩坑 1。

另外还验证了一件事：我从**原始图**和**预处理图**分别量化，张量名对齐数完全一样（65/100、17/52）。**证明名字错位是量化器本身的行为，跟预处理无关。** 这个对照实验很重要，否则会误判原因。

### 坑 5：batch 冲突（约 10 分钟）

见 3.5。修法是读图声明的形状，而不是假设。

### 坑 6：matplotlib 字体缓存（约 10 分钟）

`fontlist-v330.json.matplotlib-lock` 写系统临时目录被拒。解法是设 `MPLCONFIGDIR` 到工作区，**并且必须在 import matplotlib 之前设**。

### 坑 7：中文路径（约 5 分钟）

`cv2.imread` 读不了中文路径，静默返回 `None`。改用 `np.fromfile` + `cv2.imdecode`。

---

## 七、动手练习（重要：不做这些你等于没学）

下面每条都要求你**真的打开脚本、改代码、跑、看输出**。做完在 `NOTES.md` 里记一句。

### 练习 1：看图和跑通（30 分钟）
```powershell
cd C:\Users\xzjdtm\Documents\embedded-ai-sprint
$env:PYTHONPATH="$PWD\.pylibs"
$PY="D:\miniconda3\envs\rknn-yolov8-train\python.exe"

# 看图的统计，不看曲线
& $PY src\w1_quant\01_export_onnx.py --model squeezenet1_1 --opset 13
```
**回答**：SqueezeNet 删了多少个节点？和 MobileNetV2 的 70 个比，说明什么？

### 练习 2：证明算子融合的收益（20 分钟）
```powershell
& $PY src\w1_quant\08_ort_inference.py --benchmark --runs 30
```
**改代码**：把 `levels` 列表里的 `ORT_ENABLE_ALL` 删掉，只跑前三个级别。重跑。
**回答**：为什么跳变发生在 `EXTENDED → ALL` 之间，而不是 `DISABLE → BASIC`？

### 练习 3：亲手复现"漏层"这个坑（40 分钟）★ 最推荐

**完整步骤在 `LAB.md` 第 4.3 节**，这里只写要点。

改 `smoke_test_names.py` 第 111 行，从"全算子交集"换成"只取 Conv 输出"，
**记得先 `Remove-Item artifacts\instrumented\*_shared.onnx -Force` 清缓存**，
否则脚本会复用旧文件、改动不生效（我第一次就踩了这个）。

**回答**：
1. 两种方式下**重叠张量**的 cosine 数值有差别吗？这个事实说明了什么？
2. 对照方式漏掉的那 11 层是什么算子？为什么漏掉它们后果严重？
3. 我最初的假设（"语义错位"）为什么是错的？

### 练习 4：制造一次量化掉点（40 分钟）
把校准集换成**故意不均衡**的（比如全部用 JPEG 里色调最暗的 200 张），重跑量化。
```powershell
& $PY src\w1_quant\03_quantize.py --limit 20 --force
& $PY src\w1_quant\05_accuracy.py --num-images 200
```
**回答**：只给 20 张校准图，top-1 一致率掉到多少？和 200 张比差多少？
（我实测 10 → 500 张几乎无变化，你要能解释**为什么这个模型上掉不动**。）

### 练习 5：加一个新指标（60 分钟）
给 `common.py` 加一个 `max_abs_error(a, b)`，加进 `04_layer_analysis.py` 的输出表。
**目的**：证明你能改这套代码，而不是只会跑。

### 练习 6：换一个模型（40 分钟）
```powershell
& $PY src\w1_quant\01_export_onnx.py --model resnet18 --opset 13
```
然后手动改 `03_quantize.py` 的默认 `--model` 路径，重跑量化。
**回答**：ResNet18 的 per-tensor vs per-channel 差距，比 MobileNetV2 大还是小？**为什么？**
（提示：MobileNet 大量用 depthwise conv，ResNet 用标准 3×3 conv。想想通道间动态范围差异。）

---

## 八、面试问答（用实测数字答）

**Q: 量化为什么会掉精度？**
两个来源：权重舍入（最大 scale/2）和激活截断（超出校准观测范围的值被夹断）。第二条是校准集存在的理由。

**Q: 怎么定位是哪一层崩的？**
把两份图的中间张量都提升成图输出，逐层算 cosine 和 SQNR。实测 per-tensor 从第 0 层就跌破 0.99，最差层 cosine 0.8387 / SQNR 5.29 dB；层序号与 cosine 相关系数 -0.72，说明误差随深度累积。

**Q: 这里有什么坑？**
我一开始只取 `Conv` 输出做逐层对比，只有 17 层能对齐。我先假设是 QDQ 量化器算子融合导致语义错位，但做了对照实验后发现：两种选层方式下**重叠张量的 cosine 完全相同**，说明语义是一致的——我的假设被证伪了。真实问题是我漏掉了 10 个残差 `Add` 和池化头，**丢了三分之一的观测层**。改成"全算子候选 + 张量名交集"后拿到 29 层。

**Q: per-channel 为什么更准？代价是什么？**
权重每个输出通道一个 scale。当通道间动态范围差异大时（实测 MobileNetV2 的 classifier 层通道间 scale 差 2.9 倍），per-tensor 的单一 scale 会牺牲大部分通道。实测：top-1 一致率 0.84 → 0.98，KL 降 21 倍；代价只有 +0.24 MB（+7%）。注意 per-channel 只作用于权重，激活仍是 per-tensor（激活范围依赖输入数据，无法按通道预先确定）。

**Q: 校准集怎么选？**
估计每层激活的动态范围，不需要标签。实测在本模型上**数量几乎无收益**（10 → 500 张，mean cosine 只变 0.00084）。多样性方面我的实验结论和教科书相反，原因是评估集分布有偏 + 样本量小，报告里写了验证方向。

**Q: 什么时候必须 QAT？**
PTQ 掉点超出可接受范围，且掉点集中在少数敏感层、用混合量化也救不回来时。我们这次 per-channel PTQ 已经把 top-1 一致率做到 0.98，不需要 QAT。

**Q: INT8 一定比 FP32 快吗？**
不一定。取决于硬件有没有对应的整数指令。比如 ARMv7 没有 dot-product 指令，INT8 反而可能更慢——这是 W2 要在 i.MX6ULL 上实测的。

**Q: warmup 为什么必要？收益多大？**
必须做，但收益取决于后端。CPU + 小模型实测只有 1.02 倍；CUDA/TensorRT 因为要 autotune kernel，首帧可能慢 10 倍以上。另外做 p95/p99 统计时，第一次的异常值会污染分位数。

**Q: 端到端延迟由哪几段组成？**
采集 → 预处理（resize/归一化）→ 推理 → 后处理（NMS 等）→ 输出。本次只测了推理段（12.37ms @ 4 线程），端到端拆分是 W3 要做的事。

---

## 九、下一步（W2 会接上什么）

这份工作的结论直接指向 W2 的三个实验：

| W1 的发现 | W2 要验证什么 |
|---|---|
| 误差集中在 depthwise conv 和残差 Add | **混合量化**：把这些层留 FP16，看掉点恢复多少 |
| 校准集构成在 per-channel 下无关紧要 | 在 **per-tensor** 配置下重跑，验证"代表性"是否变成主要矛盾 |
| 模型 13.99 MB → 3.89 MB | RKNN 的 `eval_memory()` 在板端是多少；`accuracy_analysis()` 的逐层结论和 ONNX Runtime 是否一致 |
| CPU 推理 12.37ms | RK3566 NPU 上同样模型是多少；INT8 在 ARMv7（i.MX6ULL）上是否反而变慢 |
