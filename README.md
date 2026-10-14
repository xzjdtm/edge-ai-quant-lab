# 端侧 AI 部署冲刺（21 天）

目标岗位：AI 模型部署工程师 / 端侧 AI 优化工程师 / 嵌入式 AI 软件工程师。
核心目的：把简历上「端侧 AI 部署」那一栏，从**纸面承诺**变成**能扛住 3 层追问的实证**。

---

## 先读这三份

| 文件 | 作用 |
|---|---|
| [LAB.md](LAB.md) | **从这里开始**。陪练实验室：一课一个动作，告诉你敲什么、看到什么、这一步在干什么 |
| [WALKTHROUGH.md](WALKTHROUGH.md) | 原理详解。每一步为什么这么干、踩了什么坑、怎么排查 |
| [DIAGNOSIS.md](DIAGNOSIS.md) | 简历承诺 vs 真实能力的 7 处风险 + 修正后的 21 天路线 |

补充：[PROGRESS.md](PROGRESS.md)（进度 + 关键数字 + 面试自测）、[ENVIRONMENT.md](ENVIRONMENT.md)（环境与网络坑）、[GITHUB_GUIDE.md](GITHUB_GUIDE.md)（提交到 GitHub）、[NOTES.md](NOTES.md)（你的笔记）、[BOARD_DECISION.md](BOARD_DECISION.md)（板子选型）、[DECISIONS.md](DECISIONS.md)（决策记录）

---

## 当前状态

```
阶段 0  环境与决策          ✅ 完成（Linux 环境待搭，W2 前即可）
阶段 1  W1 量化误差分析      ✅ 完成，产出 2 图 1 表 + 报告
阶段 2  W2 RKNN 工具链闭环   ⬜ 未开始
阶段 3  W3 板端部署+集成     ⬜ 未开始
并行线  C++ 最小必要集       ⬜ 未开始  ← 你的真实瓶颈
```

## W1 成果

[MobileNetV2 INT8 量化误差分析报告](W1_报告/README.md)

两个核心结论（有实测数据支撑）：

1. **per-channel 是决定性因素**：top-1 一致率 0.8400 → 0.9800，KL 散度降 21 倍，代价只有 +0.24 MB。
2. **校准集数量几乎不重要**：10 张 → 500 张，mean cosine 只变 0.00084。
   而"多样性"的教科书结论在本实验里**落空了**，报告第 6.3 节如实说明了原因和验证方向。

---

## 目录结构

```
embedded-ai-sprint/
├── DIAGNOSIS.md / PROGRESS.md / ENVIRONMENT.md / BOARD_DECISION.md / DECISIONS.md
├── run_w1.ps1              一键重跑整条 W1 流水线
├── cleanup.ps1             清理沙箱锁死的临时目录
├── src/w1_quant/           W1 全部脚本（见 ENVIRONMENT.md 第 6 节）
├── tools/                  fetch_wheel.py（绕过 pip）、环境探测
├── data/calib/             校准集清单（10/50/200/500 张 + 窄域/混合/域均衡）
├── data/imagenet_sample/   1994 张真实照片，997 类
├── artifacts/              中间产物：模型 / 表格 / JSON 报告
├── W1_报告/                交付物：README.md + 2 图 + 1 表
├── .pylibs/                工作区内的 onnxruntime（不改系统环境）
└── .cache/                 权重、数据集、临时目录
```

---

## 重跑

```powershell
# 整条流水线（图片已下载时加 -SkipFetch）
powershell -ExecutionPolicy Bypass -File run_w1.ps1

# 只跑某个实验（在 embedded-ai-sprint 目录下）
$env:PYTHONPATH = "$PWD\.pylibs"
$PY = "D:\miniconda3\envs\rknn-yolov8-train\python.exe"
& $PY src\w1_quant\04_layer_analysis.py --num-images 32
& $PY src\w1_quant\07_make_report.py
```

产出的模型和表格都在 `artifacts/`，脚本默认跳过已存在的产物，加 `--force` 才会重跑。

---

## 三条不能忘的纪律

1. **每天固定 1 小时 C++**。这是你从"会跑"到"能改"的分界线，也是 ncnn 那条简历能否站住的前提。
2. **`E:\work\yolo-vision-pipeline-rknn` 不能进作品集**。它是 fork，本地零修改，面试官点开 GitHub 就能看出来。当参考可以，交付物必须自己重写。
3. **每个结论都要有数字**。没有数字的"我做过量化"在面试里等于没做。
