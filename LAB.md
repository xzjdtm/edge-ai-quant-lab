# 陪练实验室

用法：
- **一次只做一课**。做完回答「检查点」再往下，答不上就问我。
- 每课都有「这一步在干什么」——那才是要记住的东西，命令只是手段。
- 命令可以直接复制。PowerShell 里粘贴用**鼠标右键**，或 `Ctrl+V`。

所有课的起点都是同一个目录：

```powershell
cd C:\Users\xzjdtm\Documents\embedded-ai-sprint
```

---

# 第 0 课：认识你的工具（10 分钟）

## 0.1 为什么要先干这个

后面所有工作都在命令行里做。你之前用 Keil、Vivado、CubeMX 都是点鼠标，但端侧部署这套东西**没有 GUI**——交叉编译、量化、部署全是命令。所以先把命令行的基本动作练熟，否则后面每一条命令你都是"复制粘贴但不知道发生了什么"。

## 0.2 四个动作

### ① 看自己在哪
```powershell
pwd
```
**你会看到**：`C:\Users\xzjdtm\Documents\embedded-ai-sprint`

`pwd` = print working directory。**命令行里"你在哪"决定了相对路径从哪算起**，这是新手 90% 报错的原因。你输入 `src\w1_quant` 时，它是从当前目录往下的意思。

### ② 看这个目录里有什么
```powershell
ls -Name
```
**你会看到**：一列**干净的文件名/目录名**（`README.md`、`src`、`artifacts`、`data`……）。

`ls` = list，全名是 `Get-ChildItem`。`-Name` 表示"只要名字，别把大小、时间、权限那些属性都打出来"。

> **不加 `-Name` 会是什么样**：PowerShell 会把每个文件当成一个对象，打印出
> `Name / FullName / Length / CreationTime / Attributes` 等一堆属性，
> 一屏只能看两三个文件。这是 PowerShell 和 CMD 的重要区别——
> **它传的是对象不是文本**。所以任何"看列表"的命令都建议加 `-Name`。

只看目录（不看文件）：
```powershell
ls -Directory -Name
```

> **注意一个和 Linux 不一样的地方**：PowerShell 的 `ls` **默认就会列出隐藏文件**
> （Linux 下要 `ls -a` 才看得到）。所以这里没有"必须加参数才能看到点文件"这回事。

### ③ 进目录 / 退出来
```powershell
cd src\w1_quant      # 进入
ls -Name             # 看里面的脚本
cd ..                # 退回上一层
pwd                  # 确认回到原处（应显示 ...\embedded-ai-sprint）
```

`cd` = change directory，`..` 代表上一层。`cd ..\..` 退两层。

### ④ 重复上一条命令
按 **↑** 上箭头。命令行会记住历史。一直按可以翻到很早以前的命令。
**↓** 往回翻。改一条长命令时，用 ↑ 调出来再改，比重新敲快得多。

## 0.3 一个必会的技巧：Tab 补全

输入 `cd src\w1` 然后按 **Tab**，它会自动补成 `cd src\w1_quant\`。
输入文件名时也一样。**这能避免绝大部分拼写错误。**

连按两次 Tab：如果有多个匹配项，会把它们全列出来给你选。

## 0.4 检查点

回答下面两个问题（答不上就问我）：

1. 如果 `pwd` 显示的是 `C:\Users\xzjdtm`，而你输入 `ls src\w1_quant -Name`，会发生什么？
2. `ls -Name` 和直接 `ls` 的输出差别很大。**为什么 PowerShell 默认要打印那么多属性？**
   （提示：想一想"如果我只想过滤出大于 1 MB 的文件，哪种输出更有用"）

---

# 第 1 课：这个仓库里有什么（10 分钟）

## 1.1 这一步在干什么

在跑任何代码之前，先搞清楚**文件是怎么分类的**。一个能拿去面试的仓库，结构本身就是能力证明——面试官点开第一眼看的就是目录结构。

## 1.2 看结构

```powershell
ls -Name
```

现在逐个理解每个条目**为什么在这里**：

| 条目 | 是什么 | 为什么这么放 |
|---|---|---|
| `src/w1_quant/` | 所有可执行脚本 | **src = source，源码。** 脚本按执行顺序编号（01_、02_…），一眼看出流水线顺序 |
| `tools/` | 辅助工具 | `fetch_wheel.py` 是装包用的，不属于实验本身，所以单独放 |
| `data/` | 输入数据 | 图片和校准集清单。**注意它被 git 忽略了**（太大） |
| `artifacts/` | 中间产物 | 模型、表格、JSON 报告。可分两类：`tables/`+`reports/` 是**结论**（要进 git），`models/` 是**大文件**（不进 git） |
| `W1_报告/` | 交付物 | 给人看的最终成果：报告 + 图 + 表 |
| `*.md` | 文档 | 每个 md 都有明确用途，看下一节的表 |
| `.gitignore` | git 规则 | 告诉 git 哪些文件不要提交 |
| `.pylibs/` `.cache/` | 本地环境 | 装好的包和缓存，**任何机器都能重建，所以不进 git** |

## 1.3 看文档各管什么

```powershell
ls *.md -Name
```

| 文件 | 什么时候看 |
|---|---|
| `README.md` | **第一次进来**。项目的门面，GitHub 首页显示的就是它 |
| `WALKTHROUGH.md` | 想搞懂原理时。**最厚的一份** |
| `GITHUB_GUIDE.md` | 提交代码时 |
| `PROGRESS.md` | 想看进度和关键数字时 |
| `NOTES.md` | 你自己记笔记的地方 |
| `DIAGNOSIS.md` | 想知道"为什么做这件事"时 |
| `ENVIRONMENT.md` | 环境出问题时 |
| `LAB.md`（本文件） | 现在 |

> **注意这个习惯**：仓库里每个文件都有单一职责，没有"杂项.md"。
> 你自己以后写的项目也应该这样。

## 1.4 检查点

1. `artifacts/tables/` 里的 CSV 和 `artifacts/models/` 里的 ONNX，**为什么一个进 git 一个不进**？
   （提示：想一下"能不能重建"）
2. `src/w1_quant/` 里为什么脚本名要带数字前缀？

---

# 第 2 课：git 到底在干什么（15 分钟）

## 2.1 这一步在干什么

你会看到一堆 `git` 命令。**关键是理解它解决什么问题**：

> 你改了代码，跑出来的结果不对。你想回到"昨天那版"。没有版本控制，你只能靠 `代码_v2_最终版_真的最终.py`。有 git，一条命令回去。

git 保存的不是文件，是**快照历史**。

## 2.2 三个区域，这是理解 git 的关键

```
   你的文件夹            暂存区              版本库
   (working dir)      (staging area)       (repository)
        │                   │                   │
        │   git add         │   git commit      │
        ├──────────────────►├──────────────────►│
        │                   │                   │
        │◄──────────────────┴───────────────────┤
        │            git checkout               │
```

- **工作区**：你实际在编辑的文件
- **暂存区**：你"挑出来准备提交"的改动
- **版本库**：已经存下来的历史快照

**为什么要中间那个暂存区？** 因为一次提交应该只讲一件事。你改了 3 个文件，
但其中 2 个是修 bug、1 个是加功能——暂存区让你分开提交。

## 2.3 看当前状态

```powershell
git status
```

**你会看到**：一长串绿色（或蓝色）的 `new file:` / `新增文件:` 列表。

绿色 = 已经在暂存区（我帮你 `git add` 过了）。

现在看简化版：
```powershell
git status --short
```
**你会看到**每行开头有个字母：
- `A` = Added（新增，已进暂存区）
- `M` = Modified（改过，未进暂存区）
- `??` = 未跟踪（git 完全不认识它）

## 2.4 看是什么被忽略了（这一步很关键）

```powershell
git status --ignored --short | Select-String "^!!"
```

**你会看到**一堆 `!!` 开头的条目：`data/`、`.pylibs/`、`artifacts/models/`……

`!!` = 被 `.gitignore` 排除了。**这就是为什么 445 MB 的目录里只有 0.28 MB 会进 git。**

验证一下：
```powershell
git check-ignore -v artifacts/models/mobilenet_v2_opset13_sim.onnx
```
**你会看到**：类似 `artifacts/models/*.onnx` 加一个行号——它在告诉你**是 .gitignore 的哪一行**忽略了这个文件。

> **这个命令要记住**：以后你发现"文件传不上 GitHub"，第一件事就是 `git check-ignore -v <文件>`。

## 2.5 真的动手改一次（最重要的部分）

现在我们制造一个改动，走完整个 `add → commit` 流程。

**① 改一个文件**：
```powershell
notepad NOTES.md
```
记事本会打开。在最上面加一行字，比如 `# 我今天开始学了`，保存，关闭。

**② 看 git 有什么反应**：
```powershell
git status --short
```
**你会看到**：`NOTES.md` 前面是 `AM`（Added 且 Modified）——因为它在暂存区里，但工作区的内容又变了。

**③ 看具体改了什么**：
```powershell
git diff NOTES.md
```
**你会看到**你的新增行，前面带绿色的 `+`。

> `git diff` 是**最常用的命令之一**。面试时如果问你"你怎么检查改动"，
> 答这个比答"我看一眼"专业得多。

**④ 把改动放进暂存区**：
```powershell
git add NOTES.md
git status --short
```
现在它只显示 `A` 了。

**⑤ 试一个会搞坏东西的命令（安全，能撤销）**：
```powershell
git restore --staged NOTES.md     # 从暂存区拿出来，但文件内容不变
git status --short
git add NOTES.md                  # 再放回去
```
`--staged` 只影响暂存区，**不动你的文件内容**。

## 2.6 检查点（这课最重要）

1. `git add` 会不会修改文件内容？
2. `git status` 和 `git status --ignored` 有什么区别？什么情况下必须用后者？
3. 我用 `git check-ignore -v` 查出 `artifacts/models/*.onnx` 被忽略。现在我想**故意**把某一个模型提交上去，该改哪里？

---

# 第 3 课：读懂导出脚本（15 分钟）

## 3.1 这一步在干什么

打开第一个正式脚本，理解"PyTorch 模型怎么变成 ONNX 文件"。

## 3.2 先看文件多大、有哪些函数

```powershell
(Get-Content src\w1_quant\01_export_onnx.py).Count
```
**你会看到**：约 210（总行数）。

> PowerShell 里没有 Linux 那个 `wc -l`，用 `(Get-Content 文件).Count` 数行数。
> 如果觉得别扭，可以自己定义一个函数：
> ```powershell
> function wc { param($f) (Get-Content $f).Count }
> wc src\w1_quant\01_export_onnx.py
> ```

**不要一次读完 210 行**，先看骨架：

```powershell
Select-String -Path src\w1_quant\01_export_onnx.py -Pattern "^def |^class " | Select-Object LineNumber,Line
```
**你会看到**函数列表：
```
LineNumber Line
--------- ----
        41 def graph_stats(onnx_path: str) -> dict:
        68 def build_model(name: str):
        79 def use_workspace_cache() -> None:
        94 def main() -> int:
```

| 函数 | 干什么 |
|---|---|
| `graph_stats` | 统计图的节点数、算子分布、是否动态 shape |
| `build_model` | 从 torchvision 拿预训练模型 |
| `use_workspace_cache` | 把权重下载缓存指到工作区（沙箱规避，见 ENVIRONMENT.md） |
| `main` | 主流程 |

**这个习惯要养成**：读陌生代码先看函数列表，就像看目录。

## 3.3 打开看主流程

```powershell
code src\w1_quant\01_export_onnx.py
```
（`code` = VS Code。如果没有就 `notepad`。）

在 `main()` 里从上往下看，你会看到这条线：

```
1. 建模型（下载预训练权重）
2. 造一个假输入 dummy = torch.randn(1, 3, 224, 224)
3. 先用 PyTorch 跑一次，记下结果 ref        ← 基准
4. torch.onnx.export(...)                    ← 导出
5. onnx.checker.check_model(...)             ← 检查文件合法
6. onnxsim.simplify(...)                     ← 简化
7. 再用简化后的模型跑一次，和 ref 比         ← 验证没改坏
8. 打印对比表 + 存 JSON 报告
```

**第 3 步和第 7 步是这段代码的灵魂。** 没有它们，你就是在盲目相信工具。

## 3.4 亲手跑一次，而且换个模型

```powershell
$env:PYTHONPATH = "$PWD\.pylibs"
$PY = "D:\miniconda3\envs\rknn-yolov8-train\python.exe"
& $PY src\w1_quant\01_export_onnx.py --model squeezenet1_1 --opset 13
```

**你会看到**（最后几行）：
```
=== 导出前后对比 ===
  exported     nodes= ...  initializers= ...  size= ... MB  dynamic_batch=False
  simplified   nodes= ...  initializers= ...  size= ... MB  dynamic_batch=False
  onnxsim 删除了 XX 个节点 (XX.X%)
```

**记下这两组数字**，下一课要用。

## 3.5 检查点

1. 为什么代码要先用 PyTorch 跑一次（第 3 步），而不是直接导出后就相信 ONNX 是对的？
2. `simplified` 那行的 `size` 和 `exported` 那行的 `size` 比，谁大？（提示：回顾 WALKTHROUGH 第 1.3 节，想想删掉的是什么）

---

# 第 4 课：看两个模型之间发生了什么（20 分钟）

## 4.1 这一步在干什么

上一课得到一个 ONNX 文件。这一课要理解**量化对它做了什么**。

## 4.2 先让代码告诉你答案

```powershell
$env:PYTHONPATH = "$PWD\.pylibs"
$PY = "D:\miniconda3\envs\rknn-yolov8-train\python.exe"
& $PY src\w1_quant\smoke_test_names.py
```

**你会看到三大段**。现在逐段读：

### 第 1 段：张量名对比
```
FP32 nodes with output : 100
INT8 nodes with output : 303
FP32 conv-like ops     : 53
INT8 conv-like ops     : 53
shared tensor names    : 65
shared conv outputs    : 17 / 53        ← 记住这个 17
conv output order same : False
```

**这里在问**：FP32 图和 INT8 图里，命名相同的张量有多少个？

- INT8 节点数从 100 涨到 303——因为插入了大量的 `QuantizeLinear`/`DequantizeLinear`
- 卷积层两边都是 53 个——**算子没变，只是输入被量化了**
- 但只有 **17 个卷积输出**名字能对上 → **这就是问题所在**

### 第 2 段：量化参数
```
QuantizeLinear nodes   : 66
per-channel ones       : 0
    input       scale=1.866e-02..1.866e-02 zp=-14.0 ch=1 axis=None
    /features/features.0/features.0.2/Clip_output_0  scale=2.353e-02.. zp=-128.0
```

**这里在问**：每一层的量化参数是什么？

- `scale=2.353e-02` 意思是：1 个整数刻度 = 0.0235
- `zp=-128.0` 意思是：实数 0 对应整数 -128（也就是 INT8 的最小值）→ 说明这层输出**全是非负的**（ReLU 之后）
- `per-channel ones : 0` ← **这里有个陷阱，见 4.4**

### 第 3 段：逐层误差
```
comparable tensors     : 29
Conv-output-only selection would give  : 17 layers
full intersection gives               : 29 layers
the 11 extra layers are          : {'Add': 10, 'GlobalAveragePool': 1}

worst 8 tensors by cosine similarity:
  /features/features.17/conv/conv.2/Conv_output_0   0.980080   14.03
  /features/features.13/conv/conv.2/Conv_output_0   0.983118   14.67
  /features/features.13/Add_output_0                0.984132   14.98
  ...
```

**这里在问**：哪层的量化误差最大？

- **29 > 17**：全算子交集比"只取 Conv 输出"多找到 11 层，**多出来的全是 `Add` 和池化头**
- 最差的层都在 `features.13` 到 `features.17`（网络后半段），而且 **`Add` 出现在最差列表里**
- 只取 Conv 的话，`Add_output_0` 那几行**根本不会出现**

## 4.3 亲手复现那个坑（本课重点）

这一段是整份材料里**最能体现"研究怎么做"**的部分。请慢一点看。

### 背景：我最初的想法是错的

我第一次写这个分析时，取层的方式很简单：**遍历图里所有 `Conv` 算子，把输出名记下来，
在 FP32 图和 INT8 图里都暴露这些名字**。

结果只有 **17/53** 个卷积输出能对上。我当时的解释是：

> "ONNX Runtime 的 QDQ 量化器会吞掉 `Clip` 节点，并把 Clip 的输出名复用给前面的 Conv。
> 所以这 17 个名字在两张图里指向**不同的计算**——FP32 那边是激活前，INT8 那边是激活后。
> 也就是说，我一直在把不同的东西比在一起，数值毫无意义。"

这个解释听起来很专业，而且当时我写进了报告。**但它是错的。**

### 怎么发现的

我做了个对照实验：**用两种选层方式跑同一份数据，比较重叠部分的数值。**

如果我的解释成立（两种方式选到的同名张量语义不同），那么**同一个张量名在两种方式下应该给出不同的 cosine**。

### 现在你来亲手做这个实验

```powershell
code src\w1_quant\smoke_test_names.py
```

**第 111 行**现在是：
```python
    shared = intersect_tensors(BASELINE, [QUANT])
```

把它**整行替换**成下面这段（"只取 Conv 输出名"的做法）：
```python
    # 【对照实验】只取 Conv 输出，不做全算子候选
    _, fp32_conv_tmp = op_signature(BASELINE)
    _, int8_conv_tmp = op_signature(QUANT)
    _int8_names = {x for _x, x in int8_conv_tmp}
    naive_names = [n for _op, n in fp32_conv_tmp if n in _int8_names]
    shared = [
        {"index": i, "tensor": n, "op_type": "Conv", "name": ""}
        for i, n in enumerate(naive_names)
    ]
```

**关键一步：必须先清掉插桩缓存**，否则脚本会复用它之前生成的文件，你的改动就不生效了
（我第一次做这个实验就踩了这个坑，看到结果一模一样还以为实验设计有问题）：

```powershell
Remove-Item artifacts\instrumented\*_shared.onnx -Force
```

然后跑：
```powershell
$env:PYTHONPATH = "$PWD\.pylibs"
$PY = "D:\miniconda3\envs\rknn-yolov8-train\python.exe"
& $PY src\w1_quant\smoke_test_names.py
```

### 对比结果

| | Conv-only（对照） | 全算子交集（正确） |
|---|---|---|
| 比较层数 | **18** | **29** |
| 多出来的层是什么 | — | 10 个 `Add` + 1 个 `GlobalAveragePool` |
| 最差 cosine | 0.980080 | 0.980080 |
| 最好 cosine | 0.998286 | 0.998286 |
| **重叠层的数值** | **完全相同** | **完全相同** |

### 这个结果说明了什么（要记住的三点）

**① 我的原解释被证伪了。**
如果那 17 个同名张量真的语义不同，两种选层方式下它们的 cosine 不可能一模一样。
**数值相同 ⟹ 语义相同。** 所以"把激活前和激活后比在一起"这个说法是错的。

**② 真实的问题更朴素：漏层。**
Conv-only 只取到 18 层，**系统性漏掉了全部 10 个残差 `Add` 和池化头**。
不是"比错了"，而是"看得不全"。

**③ 为什么漏掉 Add 不容忽视。**
全部 29 个可比层里，`Add` 有 **10 个（34%）**，且位置都在后半段。
只取 Conv 输出，就**丢了三分之一的观测层**。

⚠️ **但别把这里过度解读**（我第一版报告就错了）：
后来核对全量数据，**Add 和普通 Conv 的平均 cosine 几乎相同**——
per-channel 下 0.9880 vs 0.9879，per-tensor 下 0.9178 vs 0.9071。
所以"Add 比 Conv 更脆弱"**不成立**。Add 出现在最差榜前列是因为它**数量多、位置靠后**，
不是因为算子本身特殊。

**准确的教训是：漏层让你少看了三分之一的样本，而不是让你漏掉了最危险的层。**
（这也是为什么"统计这类判断必须有分母"——只讲"最差的是 Add"却不讲"Add 占 34%"是误导。）

### 教训

我当时的第一反应是**往深处想**（"一定是算子融合导致语义错位"），
真相却是**浅处的疏漏**（"我的筛选条件太窄"）。

调试时容易犯这个错：**越是有经验的解释，越容易让你停止验证。**
那个"Clip 被吞掉"的说法听起来越专业，我就越不会去怀疑它——
直到我真的做了对照实验。

> **面试时的正确讲法**：
> "我一开始只取 Conv 输出，发现只有 17 层能对齐。我先假设是量化器算子融合导致语义错位，
> 但做了对照实验后发现，两种选层方式下重叠张量的 cosine **完全相同**，
> 说明语义是一致的——我的假设被证伪了。真实问题是我漏掉了 10 个残差 Add 和池化头，
> 丢了三分之一的观测层。改成全算子候选 + 张量名交集后拿到 29 层。"
>
> 这段话证明的不是"我会写代码"，而是**我会用实验推翻自己的假设**——
> 这是研究能力和工程能力的分界线。

### 改回来

```powershell
git diff src\w1_quant\smoke_test_names.py      # 先看看你改了什么
git checkout src\w1_quant\smoke_test_names.py  # 恢复原始版本
Remove-Item artifacts\instrumented\*_shared.onnx -Force   # 清缓存
& $PY src\w1_quant\smoke_test_names.py                     # 确认恢复后是 29 层
```

> `git checkout <文件>` = 丢弃该文件的工作区改动。
> **注意：未提交的改动会真的丢掉**，所以先 `git diff` 确认你确实不要了。

## 4.4 检查点

1. INT8 图的节点数（303）比 FP32（100）多很多，为什么？
2. `per-channel ones : 0` 看起来像是"per-channel 没生效"。但我们明明开了 `per_channel=True`，而且实测 per-channel 效果明显更好。**怎么解释这个矛盾？**
   （提示：看看那个统计是从哪种节点里读的——`QuantizeLinear`。再想想 per-channel 到底作用在权重还是激活上。）
3. 4.3 的对照实验里，两种选层方式下**重叠张量的 cosine 完全相同**。这个事实排除了哪种可能，又指向了哪种可能？

---

# 第 5 课：改代码（30 分钟）★ 最值钱的一课

## 5.1 这一步在干什么

前面四课你都在**读和跑**。这一课你要**改**。
改过之后，这套代码才真有你的成分——面试问到"这部分是你写的吗"你能答"是，我加了 XXX"。

## 5.2 任务：给逐层分析加一个新指标

现在每个层有三个指标：`cosine`、`sqnr_db`、`rel_l2`。
我们要加第四个：**最大绝对误差** `max_abs_err`。

**为什么这个指标有用**：cosine 看方向、SQNR 看信噪比，都是"整体"指标。
但有时候是**少数几个值错得离谱**（比如某个通道饱和到 ±127），
整体平均看不出来，最大绝对误差能抓到。

## 5.3 第一步：在 common.py 里加函数

```powershell
code src\w1_quant\common.py
```

`rel_l2` 在**第 186 行**，`rms` 在**第 196 行**。
把下面这个函数插到它们之间（也就是 195 行附近）——**这样按字母/逻辑都放得合理**：

```python
def max_abs_error(reference: np.ndarray, test: np.ndarray) -> float:
    """Largest single-element deviation between two tensors.

    Complements cosine/SQNR: those are aggregate measures, so a handful of
    badly saturated elements can hide inside a good average. This is the
    "worst single value" view.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    err = np.asarray(test, dtype=np.float64).ravel() - ref
    if err.size == 0:
        return float("nan")
    return float(np.max(np.abs(err)))
```

**这段代码逐行什么意思**：
- `np.asarray(..., dtype=np.float64).ravel()` — 转 float64 再拉平成一维
  （不管原来是 4 维 `[1,320,28,28]` 还是 2 维，一律拍平成一条线）
- `test - ref` — 逐元素相减，得到误差张量
- `np.max(np.abs(err))` — 取绝对值的最大值
- `if err.size == 0` — 防空气张量，返回 NaN 而不是崩掉

保存。

## 5.4 第二步：在分析脚本里用上它（5 处改动）

```powershell
code src\w1_quant\04_layer_analysis.py
```

**改动 1 —— 第 43 行的导入语句**

原来是：
```python
from w1_quant.common import (  # noqa: E402
    cosine_similarity,
    extract_qparams,
    format_table,
    instrument_explicit,
    intersect_tensors,
    load_layer_meta,
    redirect_tempdir,
    rel_l2,
    rms,
    sqnr,
    write_csv,
    write_json,
)
```
在 `load_layer_meta,` 后面加一行 `max_abs_error,`（保持字母序）。

**改动 2 —— 第 149 行的累加字典**

原来是：
```python
    accum: dict[str, dict] = {
        name: {"cos": [], "sqnr": [], "rel": [], "absmax": [], "rms": []}
        for name in shared
    }
```
改成（加了 `"maxerr": []`）：
```python
    accum: dict[str, dict] = {
        name: {"cos": [], "sqnr": [], "rel": [], "absmax": [], "rms": [], "maxerr": []}
        for name in shared
    }
```

**改动 3 —— 第 167~168 行附近，累加的地方**

原来是：
```python
                rec["absmax"].append(float(np.max(np.abs(a))))
                rec["rms"].append(rms(a))
```
在后面加一行：
```python
                rec["maxerr"].append(max_abs_error(a, b))
```
（注意用 `a, b` 不是 `a, b` 之外的变量——`b` 是 INT8 的输出）

**改动 4 —— 第 190~191 行，汇总字典**

原来是：
```python
            "absmax": float(np.mean(rec["absmax"])),
            "rms": float(np.mean(rec["rms"])),
```
在后面加一行：
```python
            "max_abs_err": float(np.mean(rec["maxerr"])),
```

**改动 5 —— 第 414 行的列定义**

原来是：
```python
    columns = [
        "variant", "order", "op_type", "tensor", "shape",
        "cosine", "cosine_min", "sqnr_db", "rel_l2", "absmax",
        "input_cosine", "input_rel_l2",
    ]
```
改成（`"absmax",` 后面插入 `"max_abs_err",`）：
```python
    columns = [
        "variant", "order", "op_type", "tensor", "shape",
        "cosine", "cosine_min", "sqnr_db", "rel_l2", "absmax", "max_abs_err",
        "input_cosine", "input_rel_l2",
    ]
```

保存。

> **为什么要改 5 个地方**：这是"数据流"的完整链路——
> 导入 → 建容器 → 填数据 → 汇总 → 输出。
> 漏掉任何一处，指标都不会出现在最终 CSV 里。
> **面试时能说清这条链路，比会写这一行 numpy 值钱得多。**

## 5.5 第三步：跑起来验证

```powershell
$env:PYTHONPATH = "$PWD\.pylibs"
$PY = "D:\miniconda3\envs\rknn-yolov8-train\python.exe"
& $PY src\w1_quant\04_layer_analysis.py --num-images 16
```

**怎么判断成功了**：
```powershell
Select-String -Path artifacts\tables\04_layer_metrics.csv -Pattern "max_abs_err" | Select-Object -First 1
```
**如果看到** `max_abs_err` 出现在表头里 → **成功**。

再打开 CSV 看具体数值：
```powershell
Import-Csv artifacts\tables\04_layer_metrics.csv | Select-Object -First 5 variant,tensor,cosine,max_abs_err | Format-Table
```

## 5.6 检查点

1. 你加的 `max_abs_error` 和已有的 `rel_l2`，**在什么情况下会给出不同的结论**？举个例子。
2. 改完之后跑 `git status`，你会看到哪两个文件被标记为修改？
3. 如果现在 `git checkout .`，你加的代码会怎样？（**先想，别急着试**）

---

# 第 6 课：提交到 GitHub（20 分钟）

## 6.1 为什么要学这个

不只是为了传代码。**`git` 是面试里的高频考点**，而且你以后每天都要用。

## 6.2 配置身份（只需一次）

```powershell
git config user.name  "你的真名"
git config user.email "你的常用邮箱"
git config user.name
git config user.email
```

**为什么要用真名真邮箱**：GitHub 靠邮箱把 commit 关联到你的账号。
用假的 → 你的 GitHub 贡献图上一片空白 → 面试官以为你这仓库是抄的。

## 6.3 提交

```powershell
git status --short        # 先确认要提交什么
git add -A                # 把工作区所有改动放进暂存区
git status --short        # 确认都变成 A 或 M 了
```

现在提交。**commit message 的写法很重要**，第一行是标题（≤72 字符），空一行，下面是正文：

```powershell
git commit -m "W1: MobileNetV2 INT8 量化误差分析

- ONNX 导出与 onnxsim 简化（删除 70 个 Constant 节点，数值验证 max|diff|=8.1e-06）
- 7 个 PTQ 量化变体（QDQ/QOperator、per-tensor/per-channel、三种校准算法）
- 逐层误差分析：per-channel 在 29/29 层全部优于 per-tensor
- 精度对比：top-1 一致率 0.8400 -> 0.9800，KL 散度降低 21 倍
- ONNX Runtime 基准：图优化级别 2.18x，4 线程 3.17x，warmup 1.02x
- 新增 max_abs_error 逐层指标"
```

**验证提交成功**：
```powershell
git log --oneline -3
git show --stat HEAD | Select-Object -First 20
```
**你会看到**：一个 commit hash + 你的名字 + 改动文件列表。

## 6.4 建远程仓库并推送

**先在浏览器里**：打开 https://github.com/new
- Repository name: `edge-ai-quant-lab`
- **Public**（公开。实习投递时面试官要能点开）
- **不要勾** "Add a README" / ".gitignore" / "license"
- Create

**然后回到命令行**（把 `你的用户名` 换成真的）：
```powershell
git branch -M main
git remote add origin https://github.com/你的用户名/edge-ai-quant-lab.git
git remote -v          # 确认关联正确
git push -u origin main
```

**如果要求输入密码**：输 GitHub 的 **Personal Access Token**（不是账号密码）
- 去 https://github.com/settings/tokens → Generate new token (classic)
- 勾 `repo` 权限，生成后**立刻复制**（刷新就看不到了）
- Username 填你的 GitHub 用户名，Password 粘贴 token

## 6.5 推完之后验证

```powershell
git log --oneline
git remote -v
```
**然后打开浏览器**，访问你的仓库页面，确认：
- README 显示正常（中文不乱码）
- 点进 `W1_报告/` 能看到两张图
- 文件数量约 45 个（**不是**几百个——说明 .gitignore 生效了）

## 6.6 检查点

1. `git push` 和 `git commit` 有什么区别？
2. 如果你忘了配 `user.email`，提交会怎样？GitHub 上会显示什么？
3. 仓库页面上应该有多少文件？如果看到 `data/imagenet_sample/` 里有 997 个文件，说明什么出了问题？

---

# 卡住了怎么办

## 先说一个必然出现的"假报错"

你运行 `git status --ignored` 或任何 git 命令时，可能会看到一屏红色警告：

```
warning: could not open directory '.cache/tmp/ort.quant.aivvql3w/': Permission denied
warning: could not open directory '.piptmp/pip-unpack-2ngp80wl/': Permission denied
...
```

**这是正常的，不是你的错，也不影响任何操作。**

原因：沙箱把那些临时目录的权限锁死了，git 想进去扫描但进不去。
这些目录里全是空的（是 onnxruntime/pip 用剩的），git 看不到也不影响。

想减少这些噪音可以删掉它们：
```powershell
powershell -ExecutionPolicy Bypass -File cleanup.ps1
```
（删不掉的会列出来，忽略即可。）

## 常见错误对照表

| 症状 | 原因 | 怎么修 |
|---|---|---|
| `ModuleNotFoundError: No module named 'onnxruntime'` | 没设 `PYTHONPATH` | 先跑 `$env:PYTHONPATH = "$PWD\.pylibs"` |
| `找不到路径` 或 `The system cannot find the path specified` | 当前目录不对 | `pwd` 看你在哪，`cd C:\Users\xzjdtm\Documents\embedded-ai-sprint` |
| `无法将"xxx"项识别为 cmdlet` | 命令拼错了，或者用了 Linux 的命令 | 查上面的表，`wc`/`cat`/`ls -a` 在 PowerShell 里都不同 |
| git 说没有变化 | 你改的是被忽略的文件 | `git status --short` 看清；`git check-ignore -v <文件>` 查原因 |
| `Permission denied` 一堆 | 见上面，正常现象 | 忽略，或跑 `cleanup.ps1` |
| `INVALID_ARGUMENT: Got: 4 Expected: 1` | 模型是固定 batch=1，你喂了 4 张 | 见 WALKTHROUGH 3.5 节的 `effective_batch` |

## 撤销操作速查

| 想干什么 | 命令 | 危险程度 |
|---|---|---|
| 撤销某个文件的改动 | `git checkout <文件>` | ⚠️ 该文件的改动会丢 |
| 撤销所有未提交改动 | `git checkout .` | ⚠️⚠️ 全部丢，**先 `git status` 看清** |
| 撤销最近一次 commit，保留改动 | `git reset --soft HEAD~1` | 安全 |
| 撤销 `git add`，文件不动 | `git restore --staged <文件>` | 安全 |
| 看某个文件改了什么 | `git diff <文件>` | 安全，只读 |

> **规则**：任何带 `checkout` 或 `reset --hard` 的命令，执行前先 `git status` 和 `git diff`。
> 养成这个习惯，你永远不会丢工作。

## 还是搞不定

**直接把完整报错贴给我**（从 `PS C:\...>` 那一行开始复制到最后）。
报错里的路径和行号通常就直接指向问题了。

---

# 进度

| 课 | 内容 | 状态 |
|---|---|---|
| 0 | 命令行四个动作 + Tab 补全 | ⬜ |
| 1 | 仓库结构与文件分类 | ⬜ |
| 2 | git 三个区域 + add/commit/diff/ignore | ⬜ |
| 3 | 读懂导出脚本 + 换模型跑 | ⬜ |
| 4 | 显式复现"名字错位"坑 | ⬜ |
| 5 | ★ 自己加 max_abs_error 指标 | ⬜ |
| 6 | 提交到 GitHub | ⬜ |
