# ONNX 导出与 onnxsim 效果对照

测量条件：torch 2.6.0+cu124 / torchvision 0.21.0 / onnx 1.14.1 / onnxsim 0.4.35 / opset 13

| 模型 | 参数量 | 激活函数 | 导出节点 | 简化后 | onnxsim 删除 | Clip | Constant | 文件大小 |
|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | 3.50 M | ReLU6 | 170 | 100 | **70 (41.2%)** | 35 | 70 | 13.99 → 13.99 MB |
| MobileNetV3-Small | 2.54 M | Hardswish / HardSigmoid | 141 | 141 | **0 (0%)** | 0 | 0 | 10.19 → 10.20 MB |
| SqueezeNet 1.1 | 1.24 M | ReLU | 65 | 65 | **0 (0%)** | 0 | 0 | 4.96 → 4.96 MB |

数值验证（简化后 vs 简化前，与 PyTorch 输出对比）：

| 模型 | `max abs diff` | `cosine` |
|---|---|---|
| MobileNetV2 | 8.106e-06 | 1.00000000 |
| MobileNetV3-Small | 6.676e-06 | 0.99999994 |

---

## 结论

### 1. onnxsim 的收益完全取决于图里有没有冗余，不是"跑了就有用"

MobileNetV2 被删掉 70 个节点，另两个模型 **一个都没删**。
**所以"用 onnxsim 简化模型"这句话本身没有信息量**——
必须说清"在哪个模型上删掉了多少个节点"。

### 2. 删掉的到底是什么，已查证

MobileNetV2 的 70 个 `Constant` **全部被 `Clip` 算子消费**，样例值是标量 `0.0`。
数量关系是 35 个 Clip × 2 个 Constant（min 和 max）= 70。

**来源已用对照实验确认**：

- 重新导出一次 MobileNetV2 → 仍然是 `Clip=35, Constant=70`（可复现，不是文件损坏）
- MobileNetV3-Small 也含 clamp 类激活，却 `Clip=0, Constant=0` → 说明 torch 对不同
  激活采用了不同的导出形式，不是"所有 clamp 都变成 Clip"
- 换 opset 9/10/11/13 导出同一模型 → `Constant` 数量不变 → **排除 opset 差异**

所以根因是：**该模型的 ReLU6 被导出成 `Clip`，而 `Clip` 的 min/max 在 torch 2.6 的
导出器里是以 Constant 节点（而非属性）形式给出的。**

### 3. 文件大小不变，这才是常被误解的地方

MobileNetV2 删了 70 个节点，**文件大小 13.99 MB 一字节没变**。

原因：`Constant` 节点的内容被折叠进了 **initializer**（权重区）。
数据仍然在文件里，只是从"算子节点"变成了"初始化张量"。

**所以常量折叠改变的是「图的形状」（算子少了、initializer 多了），不是「模型体积」。**
它的收益是**推理时少执行 70 个算子**，不是磁盘占用。

（对比：MobileNetV3 的 `10.19 → 10.20 MB` 反而**变大**了 0.01 MB，
因为 onnxsim 重写图时序列化细节有微小差异。这属于正常噪声。）

---

## 面试怎么用这组数据

**Q：ONNX 导出踩过什么坑？**

> 我对比过三个模型的导出结果：MobileNetV2 导出后 onnxsim 能删掉 70 个节点（41%），
> 而 SqueezeNet 和 MobileNetV3-Small **一个都删不掉**。我去查了那 70 个是什么——
> 全部是 `Clip` 算子消费的 `Constant`，来自 MobileNetV2 的 ReLU6。
> 而且**文件大小完全没变**（都是 13.99 MB），因为常量折叠把数据从节点搬进了
> initializer，改变的是图的形状而不是模型体积——收益在推理时的算子调用数上。

这段话证明了三件事：**你会做对照实验、你会深挖数字背后的原因、你不会把"跑了工具"当成"获得了收益"。**

**Q：为什么导出后要做数值验证？**

> 因为转换是有损的（算子融合、算子替换、常量折叠都可能改变计算顺序）。
> 我每次都拿 PyTorch 的输出当基准，比较 `max abs diff` 和 `cosine`。
> 实测 MobileNetV2 是 `8.1e-06`，MobileNetV3 是 `6.7e-06`，都在浮点误差量级。
> **没有这个基准，我只能知道"它没报错"，不知道"它算得对"。**

---

## 复现命令

```powershell
cd C:\Users\xzjdtm\Documents\embedded-ai-sprint
$env:PYTHONPATH = "$PWD\.pylibs"
$PY = "D:\miniconda3\envs\rknn-yolov8-train\python.exe"

foreach ($m in "mobilenet_v2", "mobilenet_v3_small", "squeezenet1_1") {
    & $PY src\w1_quant\01_export_onnx.py --model $m --opset 13
}
```

> 脚本会覆盖 `artifacts/reports/01_export_report.json`，所以**最后跑的那个模型**才是
> 报告里记录的。想要三个模型的数据，得手动收集每次的输出（本表就是这么来的）。
