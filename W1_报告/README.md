# MobileNetV2 INT8 量化误差分析

> 一周实验的完整记录：环境 / 导出 / 量化 / 逐层误差 / 校准集影响。

## 1. ONNX 导出

| 阶段 | 节点数 | initializer | 大小(MB) | 动态 batch |
|---|---|---|---|---|
| exported | 170 | 106 | 13.99 | False |
| simplified | 100 | 108 | 13.99 | False |

- onnxsim 后数值验证：`max|diff| = 8.106e-06`，`cosine = 1.00000000`
- onnxsim 删除 70 个节点（41.2%），节点数 170 → 100；同时 initializer 从 106 增加到 108。实测删除的全部是 **`Constant` 算子**（70 个），它们被折叠成了 initializer。
- 注意：文件大小**没有变化**（13.99 MB）。所以常量折叠改变的是**图的形状**（节点少了、initializer 多了），不是模型体积——推理时少执行 70 个算子，权重字节一个没少。

## 2. 量化变体

| 变体 | 格式 | per-channel | 校准方法 | 大小(MB) | Q | DQ | QLinear |
|---|---|---|---|---|---|---|---|
| qdq_pertensor_minmax | QDQ | False | MinMax | 3.65 | 66 | 172 | 0 |
| qdq_perchannel_minmax | QDQ | True | MinMax | 3.89 | 66 | 172 | 0 |
| qdq_perchannel_entropy | QDQ | True | Entropy | 3.89 | 66 | 172 | 0 |
| qdq_perchannel_percentile | QDQ | True | Percentile | 3.89 | 66 | 172 | 0 |
| qoperator_pertensor | QOperator | False | MinMax | 3.61 | 2 | 2 | 52 |
| qoperator_perchannel | QOperator | True | MinMax | 3.70 | 2 | 2 | 52 |
| qdq_perchannel_narrowcalib | QDQ | True | MinMax | 3.89 | 66 | 172 | 0 |

## 3. 逐层误差分析

### qdq_pertensor_minmax

- 对比张量数：29
- 平均 cosine：0.906162
- 最差层：`/features/features.16/conv/conv.2/Conv_output_0`（Conv，第 24 层），cosine=0.838729，SQNR=5.29 dB
- **第一个跌破 0.99 的层**：`/features/features.1/conv/conv.1/Conv_output_0`（Conv，第 0 层），cosine=0.988701

### qdq_perchannel_minmax

- 对比张量数：29
- 平均 cosine：0.988160
- 最差层：`/features/features.13/conv/conv.2/Conv_output_0`（Conv，第 19 层），cosine=0.979623，SQNR=14.78 dB
- **第一个跌破 0.99 的层**：`/features/features.6/conv/conv.2/Conv_output_0`（Conv，第 7 层），cosine=0.987176

## 4. 精度对比


模型：MobileNetV2（ImageNet 预训练，ONNX opset 13 + onnxsim）
量化：ONNX Runtime `quantize_static`，权重与激活均为 INT8

| 变体 | 格式 | per-channel | 校准方法 | 大小(MB) | top1一致率 | 置信度下降 | KL散度 | 最大logit偏差 |
|---|---|---|---|---|---|---|---|---|
| qdq_pertensor_minmax | QDQ | False | MinMax | 3.65 | 0.8400 | 0.1272 | 0.19393 | 4.9483 |
| qdq_perchannel_minmax | QDQ | True | MinMax | 3.89 | 0.9800 | 0.0116 | 0.00910 | 0.9208 |
| qdq_perchannel_entropy | QDQ | True | Entropy | 3.89 | 0.9800 | 0.0116 | 0.00910 | 0.9208 |
| qdq_perchannel_percentile | QDQ | True | Percentile | 3.89 | 0.9800 | 0.0116 | 0.00784 | 0.8603 |
| qoperator_pertensor | QOperator | False | MinMax | 3.61 | 0.8500 | 0.1288 | 0.19497 | 5.0182 |
| qoperator_perchannel | QOperator | True | MinMax | 3.70 | 0.9800 | 0.0114 | 0.00900 | 0.9208 |
| qdq_perchannel_narrowcalib | QDQ | True | MinMax | 3.89 | 0.9800 | 0.0108 | 0.00883 | 0.7705 |

说明：`top1一致率` = INT8 与 FP32 选取同一类的比例；`置信度下降` = 两者的平均 top-1 概率之差；`KL散度` 衡量整个输出分布的偏移，而不只是 argmax。

## 5. 校准集的影响

### 5.1 规模

| 校准图片数 | mean cosine | min cosine | SQNR(dB) | top1一致率 | KL | 置信度下降 |
|---|---|---|---|---|---|---|
| 10 | 0.99055 | 0.96785 | 18.07 | 0.9688 | 0.00642 | 0.0068 |
| 50 | 0.99045 | 0.96869 | 18.01 | 0.9375 | 0.00656 | 0.0065 |
| 200 | 0.98986 | 0.96705 | 17.77 | 0.9375 | 0.00756 | 0.0102 |
| 500 | 0.98971 | 0.96539 | 17.72 | 0.9062 | 0.00735 | 0.0070 |

### 5.2 多样性（图片总数相同）

| 校准集 | 图片数 | mean cosine | min cosine | SQNR(dB) | top1一致率 | KL |
|---|---|---|---|---|---|---|
| narrow_500 | 500 | 0.98989 | 0.96478 | 17.79 | 0.9375 | 0.00760 |
| mixed_500 | 500 | 0.98977 | 0.96554 | 17.73 | 0.9062 | 0.00863 |
| domain_balanced_500 | 500 | 0.98972 | 0.96457 | 17.70 | 0.8438 | 0.00753 |

## 6. 结论（从上面的数据直接读出来的）

### 6.1 per-channel 是这次量化的决定性因素，不是校准集

- per-tensor → per-channel：top-1 一致率 **0.8400 → 0.9800**，KL 散度 **0.19393 → 0.00910**（降低约 21 倍），最大 logit 偏差 **4.95 → 0.92**。
- 代价只有 **3.65 MB → 3.89 MB**（+7%）。用 0.24 MB 换回 14 个百分点的一致率，是端侧部署里性价比最高的一次取舍。
- 置信度下降从 **0.1272 降到 0.0116**：这一点对实际流水线很关键——argmax 一致不代表概率分布不变，下游如果按阈值做拒识或做 NMS，分布漂移会直接改变行为。

### 6.2 误差随层深累积，但两类量化的形态完全不同

- **per-tensor 从第 0 层就开始崩**（`/features/features.1/conv/conv.1/Conv_output_0`，cosine=0.9887），之后一路下滑，最差层 cosine 只有 **0.8387**。
- **per-channel 撑到第 7 层**才跌破阈值，最差层 cosine 仍有 **0.9796**，整条曲线基本贴着 0.98–0.99，没有崩溃段。
- 最脆弱的层集中在中后段的 **depthwise 卷积和残差 Add**：`…eatures/features.16/conv/conv.2/Conv_output_0`、`…eatures/features.13/conv/conv.2/Conv_output_0` 一类。这与 MobileNet 系列的已知结论一致——逐深度卷积每个通道只有一个卷积核，通道间动态范围差异大，per-tensor 的单一 scale 必然两头不讨好。

### 6.3 校准集：数量几乎不重要，而"多样性"的结论是反直觉的

- 校准集从 **10 张 → 500 张**（50 倍），mean cosine 只从 0.99055 变到 0.98971（变化 -0.00084），SQNR 反而下降 0.35 dB。**在这个模型上，堆校准图片数量几乎没有任何收益。**
- 同样 500 张，三种构成的差异：`narrow_500`=0.9375、`mixed_500`=0.9062、`domain_balanced_500`=0.8438。**全部落在噪声量级。**
- 先看评估集的真实分布（这个数据决定了怎么解读）：全部 1994 张图按类别 ID 排序后，**评估集取的尾部 64 张全部属于 object/scene 域**，而 `domain_animal` 组只占整体的一小部分。
- **预期落空的地方要老实说**：教科书结论是"校准集要代表部署分布"，按此 `domain_balanced_500`（三个域等量抽取）应该最好，实测它反而最低（0.8438）。而表现最好的 `narrow_500` 取自 animal 域，**和评估集所在域并不匹配**——这与"分布匹配"的直觉正好相反。
- 所以"域不匹配所以掉点"这个解释在这里**站不住**。能站住的是这个：per-channel 已经把 mean cosine 压到 0.989~0.990 的窄带内，各配置之间只差 0.0002，**校准集构成的影响已经被压到测量分辨率以下**；此时 64 张评估集上 0.906 与 0.938 的差距只相当于 2~3 张图的翻转，不足以支撑任何结论。要定论至少需要 500 张评估图并重复多次。
- 因此本节的结论不是"多样性不重要"，而是：**当量化配置（per-channel）本身已经把误差压到很低时，校准集构成退居次要；它是否成为主要矛盾，取决于量化配置有多激进。** 
- 验证方法很明确：在 **per-tensor** 配置下重跑本节——那里 mean cosine 只有 0.913，离散度是 per-channel 的 3 倍以上，校准集构成的差异才有机会从噪声里露出来。这正是 W2 混合量化要回答的问题。


## 7. 图表

![图1 逐层量化误差（cosine / SQNR）](W1_报告/fig1_layer_curve.png)

![图2 校准集规模与多样性影响](W1_报告/fig2_calib_sweep.png)

## 8. 本实验的边界（面试时主动说明，比被问出来好）

- 评估集是 **ImageNet 每个类别一张真实照片**（`EliSchwartz/imagenet-sample-images`，共约 2000 张，覆盖 997 类），来源和规模都有限，**绝对精度数字不可外推**；本实验的结论是**相对趋势**。
- 评估集按类别 ID 排序后取尾部 64 张，实测 **100% 落在 object/scene 域**，与整体分布（animal 占小部分）不一致；64 张的样本量使 top-1 一致率的分辨率约为 1/64≈1.6 个百分点。这直接限制了 5.2 的解读，见 6.3。
- 量化在 x86 CPU 上用 ONNX Runtime 完成，代表 PTQ 的一般行为；RKNN / nncase 的具体算子实现会带来差异，W2/W3 会给出 NPU 侧数据。
- `quantize_static` 的校准是 MinMax/Entropy/Percentile，不是逐层敏感度搜索；混合量化的选层策略放在 W2 做。
