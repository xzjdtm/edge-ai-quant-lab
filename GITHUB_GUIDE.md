# 提交到 GitHub

目录名是 `embedded-ai-sprint`，但**仓库名建议用 `edge-ai-quant-lab`**——目录名不影响仓库名。

---

## 先说清楚：这份代码能不能算你的

**能。** 但有三条线要分清楚，面试都问得到。

### ✅ 可以提交

- 所有代码、脚本、文档都是**为这个任务新写的**，没有从别处抄。
- 和"用 Copilot / ChatGPT 生成代码然后提交"是同一性质。工程上很常见。

### ❌ 不能提交（我们已经避开了）

- `E:\work\yolo-vision-pipeline-rknn` —— fork 自 `VaporTang`，本地零修改。
  **这个绝对不能进作品集**：面试官点开 GitHub 看到 "forked from ..." 就知道不是你的。
  当参考可以，代码必须自己重写。

### ⚠️ 真正的风险：提交了但讲不出来

面试官判断代码是不是你写的，不靠看 commit 记录，靠**追问**：

> "你这个 `intersect_tensors` 为什么用集合交集，不能用字典？"
> "如果模型有 200 层，你这个 chunk=24 会不会有问题？"
> "你把 `--threshold` 改成 0.995，结果会怎么变？"

**答不出来，代码就是负资产**——比放空仓库更糟，因为它暴露了你"会把别人的东西说成自己的"。

所以我给你写了 `WALKTHROUGH.md` 和 `NOTES.md`。**提交之前，至少做完练习 1、3、5。**
练习 3 是核心（复现那个"名字错位"的坑），练习 5 要求你**改代码加一个新指标**——
改过之后，这套代码就真有你的成分了。

**建议节奏**：先提交（记录时间点），然后做练习，边做边补 commit。
这样 commit 历史本身就是"学习过程"的证据，比一次性传 4000 行更像真的。

---

## 提交流程

### 第 1 步：配置身份（替换成你的真名和邮箱）

```powershell
cd C:\Users\xzjdtm\Documents\embedded-ai-sprint

git config user.name  "你的名字"
git config user.email "你的邮箱"

# 确认
git config user.name; git config user.email
```

> 用 `--local`（默认）只对本仓库生效，不动你的全局配置。
> **建议用真名 + 常用邮箱**，GitHub 会把它关联到你的账号贡献图。

### 第 2 步：确认只提交该提交的

```powershell
git status --short
# 应该看到 44 个文件，没有 data/、.pylibs/、*.onnx
```

如果不确定某个大文件是否被忽略：

```powershell
git check-ignore -v data/imagenet_sample .pylibs artifacts/models/mobilenet_v2_opset13_sim.onnx
```

### 第 3 步：首次提交

```powershell
git commit -m "W1: MobileNetV2 INT8 量化误差分析

- ONNX 导出与 onnxsim 简化（删除 70 个 Constant 节点，数值验证 max|diff|=8.1e-06）
- 7 个 PTQ 量化变体（QDQ/QOperator、per-tensor/per-channel、MinMax/Entropy/Percentile）
- 逐层误差分析：per-channel 在 29/29 层全部优于 per-tensor
  平均 cosine 0.9062 -> 0.9882，首个崩掉的层从第 0 层推迟到第 7 层
- 精度对比：top-1 一致率 0.8400 -> 0.9800，KL 散度降低 21 倍
- 校准集实验：规模从 10 到 500 张几乎无收益；多样性结论与预期相反，报告中说明原因
- ONNX Runtime 基准：图优化级别 2.18x，4 线程 3.17x，warmup 1.02x

产出：2 张曲线图 + 1 张对比表 + 完整报告"
```

多行 commit message：第一行是标题（≤72 字符），空一行，后面是正文。

### 第 4 步：在 GitHub 上建仓库

1. 打开 https://github.com/new
2. **Repository name**: `edge-ai-quant-lab`
3. **Description**: `MobileNetV2 INT8 quantization error analysis: per-layer drift, per-tensor vs per-channel, calibration set study (ONNX Runtime)`
4. **Public** ← 选公开。实习投递时面试官要能点开看
5. ⚠️ **不要勾** "Add a README file" / ".gitignore" / "license"
   —— 本地已经有文件了，勾了会产生冲突
6. 点 Create repository

### 第 5 步：关联远程并推送

```powershell
cd C:\Users\xzjdtm\Documents\embedded-ai-sprint

git branch -M main
git remote add origin https://github.com/<你的用户名>/edge-ai-quant-lab.git
git push -u origin main
```

### 第 6 步：认证（如果要求密码）

GitHub 已经不允许用账号密码推送，需要 **Personal Access Token**：

1. https://github.com/settings/tokens → **Generate new token (classic)**
2. Note 写 `dsh-push`，Expiration 选 90 天，勾选 **`repo`** 权限
3. 生成后**立刻复制**（页面刷新就看不到了）
4. `git push` 时：
   - Username: 你的 GitHub 用户名
   - Password: **粘贴那个 token**（不是账号密码）

> Token 会被 Windows 凭据管理器存下来，下次不用再输。
> **别把 token 写进任何代码或文档里。**

更省事的办法是装 GitHub CLI：

```powershell
winget install --id GitHub.cli
gh auth login          # 按提示走浏览器授权
```

装完之后 `git push` 会自动用 `gh` 的凭据。

---

## 提交后要检查的三件事

```powershell
# 1. 确认仓库大小合理（应该在 1 MB 以内）
git count-objects -vH

# 2. 确认大文件没进去
git ls-files | Select-String -Pattern 'onnx$|\.pt$|imagenet_sample|pylibs'

# 3. 确认远程正确
git remote -v
```

然后**打开 GitHub 网页确认**：
- 首页 README 能正常显示（中文不乱码、图片能加载）
- `W1_报告/fig1_layer_curve.png` 点开能看到曲线
- commit message 完整显示

---

## 后续 Commit 的节奏建议

不要一次性传完就不动了。真实的学习过程是渐进的，commit 历史应该反映这一点：

| 时机 | commit message 示例 |
|---|---|
| 现在 | `W1: MobileNetV2 INT8 量化误差分析` |
| 做完练习 5（加指标） | `feat: 逐层分析增加 max_abs_error 指标` |
| 做完练习 6（换模型） | `experiment: 补充 ResNet18 对照（per-tensor vs per-channel 差距更小）` |
| W2 RKNN 跑通 | `W2: 接入 RKNN Toolkit2，accuracy_analysis 逐层结论与 ORT 一致` |
| W3 板端数据 | `W3: RK3566 板端实测（NPU vs CPU），补充端到端延迟拆分` |

每次 commit **只做一件事**，message 里带数字。面试官翻你的 commit 列表，
看到的是"这个人持续在推进并且每次都记录数据"——这比 4000 行一次提交可信得多。

---

## 一个建议：加 LICENSE

公开仓库没有 LICENSE，严格说别人不能合法使用你的代码（虽然没人会追究）。
加一个 MIT 协议是 10 秒钟的事，而且显得专业：

```powershell
# 新建 LICENSE 文件，内容用 MIT 模板，把 <year> 和 <name> 换成你的
notepad LICENSE
```

然后：

```powershell
git add LICENSE
git commit -m "chore: 添加 MIT 协议"
git push
```

---

## 简历里怎么写这个项目

等你能讲清楚之后，加这样一条（**数字全部真实**）：

```
端侧模型量化误差分析（独立完成）
技术栈：Python / PyTorch / ONNX / ONNX Runtime / NumPy / Matplotlib
· 搭建 ONNX 导出→INT8 PTQ 量化→逐层误差分析流水线，覆盖 7 种量化配置
  （QDQ/QOperator、per-tensor/per-channel、MinMax/Entropy/Percentile）
· 定位量化掉点机理：通过逐层 cosine 相似度与 SQNR 分析，确认误差集中在中后段
  depthwise 卷积与残差连接；per-channel 量化使全部 29 个可比层指标提升，
  首个失效层由第 0 层推迟至第 7 层
· 量化后 top-1 一致率由 84.0% 提升至 98.0%，KL 散度降低 21 倍，模型体积
  13.99MB→3.89MB；完成 7 组量化配置的精度-体积对比实验
· 校准集研究：验证 10~500 张校准集规模对精度影响 <0.1%，并分析出评估集分布
  偏差对结论的影响，给出 per-tensor 配置下的验证方案
· 排查并解决 ONNX Runtime QDQ 量化器算子融合导致的中间张量错位问题，
  通过张量名交集匹配将可比层从 17 层（错误匹配）修正为 29 层
```

**这段的价值在哪**：前三条是"做了什么"，第四条是"数据"，第五条是"工程判断力"。
面试官最看重的往往是最后一条——因为它证明你会怀疑自己的结果。

---

## 常见追问与预案

**Q: 这个是你自己写的吗？**
A: 代码是我写的，用 AI 辅助生成了一部分。但设计、调试和结论我自己做了一遍——
（然后讲练习 3 那个坑：你怎么发现 17/53 对不齐、怎么定位到 Clip 被吞掉、怎么修）。
**承认用了 AI 辅助不是减分项，讲不出细节才是。**

**Q: 为什么选 MobileNetV2？**
A: 两个原因：① 它是 depthwise separable conv 的代表，通道间动态范围差异大，
是 per-tensor 量化最经典的失败案例；② K230 项目用的就是 MobileNet 系列，
我想搞清楚当时量化掉点的机理。

**Q: 为什么不用真实的数据集（ImageNet val）？**
A: 一开始就试了。官方源在这台机器上只有 0.02 MB/s，170MB 的 CIFAR-10 要跑 2.6 小时，
792MB 的 Oxford Pets 要 8 小时。所以改用 GitHub 上每类一张的 ImageNet 样图，
逐张下载（0.33 MB/s，断了只损失一张）。**代价是绝对精度数字不能外推，但相对趋势有效**——
这一点我写进了报告第 8 节。

**Q: 这个项目有什么不足？**
A: 三个：① 评估集只有 997 类各一张，分布有偏，影响了校准集多样性实验的结论；
② 只在 x86 CPU 上验证，没有 NPU 侧数据（这是 W2/W3 要补的）；
③ 混合量化（把敏感层留 FP16）还没做。**主动说出不足，比等面试官挖出来好。**
