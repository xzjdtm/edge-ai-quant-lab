# W1 量化对比总表

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