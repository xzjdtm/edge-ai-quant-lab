# W1 环境搭建与运行说明

目标：在一台**没有任何 Linux 环境**的 Windows 机器上，跑通
「ONNX 导出 → INT8 PTQ 量化 → 逐层误差分析 → 校准集实验 → 报告」。

---

## 0. 为什么这里有一套自己的 wheel 安装机制

这台机器上 `pip install` 会失败：

```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied:
  '...\Temp\dsh-xxxx\pip-unpack-xxxx\onnxruntime-1.17.3-cp310-cp310-win_amd64.whl'
```

两个原因叠加：

1. DSH 的文件沙箱**禁止写入 conda 目录**（`D:\miniconda3\...`）。
2. 沙箱会给每个子进程**强制注入 `TEMP=...\Temp\dsh-xxxx`**，而那个目录里的文件
   解压后读不回来。

所以本目录用 `tools/fetch_wheel.py` **绕过 pip**：直接向 PyPI 拿 wheel URL，
下载到工作区，用 `zipfile` 解压，再通过 `PYTHONPATH` 引用。
不改动任何系统环境。

---

## 1. 用哪个 Python

**必须用 `rknn-yolov8-train` 这个 conda 环境（Python 3.10）**，它已经装好了
`torch 2.6.0+cu124 / torchvision / onnx 1.14.1 / onnxsim 0.4.35 / scikit-learn`。

`base` 环境是 Python 3.14，`onnx` 没有对应 wheel（cp312 的扩展在 3.14 下 ABI 不兼容）。

```powershell
$PY = "D:\miniconda3\envs\rknn-yolov8-train\python.exe"
$WS = "C:\Users\xzjdtm\Documents\embedded-ai-sprint"
$env:PYTHONPATH = "$WS\.pylibs"     # 必需：onnxruntime 装在这里
```

## 2. 已装到工作区的包

| 包 | 版本 | 为什么需要 |
|---|---|---|
| onnxruntime | 1.17.3 | `quantize_static` / `CalibrationMethod` / QDQ 格式 |
| （onnx 1.14.1） | 系统已有 | — |

重装方式：

```powershell
cd $WS
& D:\miniconda3\python.exe tools\fetch_wheel.py onnxruntime `
    --version 1.17.3 --py cp310 --dest .pylibs `
    --index https://pypi.tuna.tsinghua.edu.cn/pypi
```

> `--py cp310` 是必需的：用 Python 3.14 解释器去下 **3.10 的 wheel**。

## 3. 网络现实（实测，别浪费时间去试）

| 源 | 实测速度 | 可用性 |
|---|---|---|
| 清华 PyPI 镜像 | **1.6 MB/s** | ✅ 可用 |
| hf-mirror.com | **1.2 MB/s** | ✅ 可用 |
| raw.githubusercontent.com | **0.33 MB/s** | ✅ 可用（小文件） |
| cs.toronto.edu（CIFAR-10 官方） | 0.02 MB/s | ❌ 不可用 |
| robots.ox.ac.uk（Oxford Pets） | 0.03 MB/s | ❌ 不可用 |
| mirror.sjtu.edu.cn `cifar/` | 5 MB 后截断 | ❌ 假镜像 |
| github.com releases | 超时 | ❌ 不可用 |

结论：**图片通过 `raw.githubusercontent.com` 一张张拿**（每张 ~100 KB，
断了只损失一张），比下整个数据集归档稳得多。

---

## 4. 运行顺序

```powershell
cd C:\Users\xzjdtm\Documents\embedded-ai-sprint
$PY = "D:\miniconda3\envs\rknn-yolov8-train\python.exe"
$env:PYTHONPATH = "$PWD\.pylibs"
$env:PYTHONUNBUFFERED = "1"

# --- 数据 ---
& $PY src\w1_quant\02b_fetch_github_images.py --download --max-classes 1000
& $PY src\w1_quant\02_prepare_calib.py --from-dir data\imagenet_sample --max-images 600

# --- 导出 ---
& $PY src\w1_quant\01_export_onnx.py --model mobilenet_v2 --opset 13

# --- 量化（会调用 quant_pre_process 生成公平基线）---
& $PY src\w1_quant\03_quantize.py

# --- 冒烟测试：确认张量名可对齐（第一次跑必做）---
& $PY src\w1_quant\smoke_test_names.py

# --- 三个实验 ---
& $PY src\w1_quant\04_layer_analysis.py --num-images 32      # 实验①逐层
& $PY src\w1_quant\05_accuracy.py --num-images 200           # 精度对比
& $PY src\w1_quant\06_calib_sweep.py --num-eval-images 64    # 实验②③校准集

# --- 出图出报告 ---
& $PY src\w1_quant\07_make_report.py
```

或者一键：

```powershell
powershell -ExecutionPolicy Bypass -File run_w1.ps1
```

---

## 5. 产出物

```
artifacts/
  models/     mobilenet_v2_opset13.onnx / _sim.onnx / _preproc.onnx
              mobilenet_v2_opset13_sim_qdq_*.onnx / qoperator_*.onnx
  tables/     03_quant_variants.csv, 04_layer_metrics.csv,
              05_accuracy.csv, 06_calib_sweep.csv
  reports/    01_export_report.json ... 06_calib_sweep.json
W1_报告/
  README.md              完整报告（面试可直接讲）
  table1_summary.md      对比表
  fig1_layer_curve.png   逐层误差曲线
  fig2_calib_sweep.png   校准集规模/多样性曲线
```

---

## 6. 脚本职责

| 脚本 | 作用 | 对应天数 |
|---|---|---|
| `tools/fetch_wheel.py` | 绕过 pip 装包 | 环境 |
| `02b_fetch_github_images.py` | 拉真实图片 | D6 前置 |
| `02_prepare_calib.py` | 造校准集（规模/域/混合） | D6 前置 |
| `01_export_onnx.py` | PyTorch→ONNX + onnxsim + 数值校验 | **D2** |
| `03_quantize.py` | PTQ 量化变体矩阵 | **D5** |
| `08_ort_inference.py` | ORT 会话/provider/IO binding/图优化/线程/基准 | **D3** |
| `smoke_test_names.py` | 验证逐层对比可行性 | D5 后 |
| `04_layer_analysis.py` | 逐层 cosine/SQNR，找第一个崩的层 | **D6①** |
| `05_accuracy.py` | top1一致率/置信度/KL | **D6** |
| `06_calib_sweep.py` | 校准集规模 + 多样性 | **D6②③** |
| `07_make_report.py` | 出图 + 出报告（含自动生成的结论段） | **D7** |
| `common.py` | 指标 / 图改写 / 量化参数提取 / 沙箱规避 | — |

---

## 7. 踩过的坑（面试素材）

### 7.1 沙箱：新建目录如果是 0700 权限，就再也写不进去

这是本次最花时间的一个坑，值得完整记录。表现是一串互相矛盾的错误：

```
PermissionError: [Errno 13] Permission denied: '...\Temp\dsh-xxxx\pip-unpack-xxxx\onnxruntime-....whl'
RuntimeError: SystemError : 13
PermissionError [WinError 5]: '...\Temp\dsh-xxxx\ort.quant.ef2y2u56'
```

排查过程与结论：

| 观察 | 结论 |
|---|---|
| 下载 15 秒就成功，只在解压时失败 | 不是网络问题 |
| 系统临时目录、工作区临时目录都失败 | 不是目录位置问题 |
| PowerShell 自己 `Remove-Item` 也报"访问被拒绝" | 是 ACL 问题，不是文件占用 |
| 100 MB 大文件在工作区写入/读回正常 | 不是大文件问题 |
| **`os.makedirs(d)` 可写，`tempfile.mkdtemp(d)` 不可写** | **`mkdtemp` 用 mode 0700** |
| `os.mkdir(d, 0o700)` 后 `os.chmod(d, 0o777)` 被拒 | 创建瞬间就被锁定，补救无效 |

所以修法是**在创建时就用默认权限**：

```python
# common.py::redirect_tempdir() 里替换掉 tempfile 的行为
def _mkdtemp(suffix="", prefix="tmp", dir=None):
    name = os.path.join(target, prefix + next(tempfile._get_candidate_names()) + suffix)
    os.makedirs(name)          # 默认权限（0777 & ~umask），可写
    return name

tempfile.mkdtemp = _mkdtemp
tempfile.TemporaryDirectory = _ResilientTemporaryDirectory   # 清理失败也不抛
```

同时 `redirect_tempdir()` 还处理了另外三件事：

- `tempfile.tempdir` 指到工作区（沙箱注入的 `TEMP` 无法通过环境变量覆盖，
  但 `tempfile` 会缓存模块属性，进程内改它就够了）
- `MPLCONFIGDIR` 指到工作区（matplotlib 的字体缓存锁文件也踩同一个坑）
- 每次启动清扫上一轮删不掉的临时目录

另外 pip 完全绕开：`tools/fetch_wheel.py` 直接拿 wheel URL、下载、`zipfile` 解压、
再用 `PYTHONPATH` 引用。**不改动任何系统环境**。

### 7.2 其他坑

1. **Python 3.14 没有可用的 onnx wheel**：`onnx 1.16.2` 的 `cp312` 扩展在 3.14 下报
   `cannot import name 'ONNX_ML'`（ABI 不兼容）。必须用 py3.10 的
   `rknn-yolov8-train` 环境。
2. **数据集镜像被限速**：cs.toronto.edu / robots.ox.ac.uk 只有 **0.02 MB/s**，
   792 MB 的 Oxford Pets 要跑 8 小时。mirror.sjtu.edu.cn 的 `cifar/` 是**假镜像**
   （给 5 MB 就断）。最终用 GitHub raw 逐张拿（0.33 MB/s，断了只损失一张）。
3. **torch 权重缓存目录**：默认写 `~/.cache/torch` 被拒 → `TORCH_HOME` 指到工作区。
4. **cv2 读不了中文路径**：用 `np.fromfile` + `cv2.imdecode` 绕过。

### 7.3 量化分析本身的坑：漏层比错配更常见（这条最值钱）

**先说结论**：逐层对比时，"只取 Conv 输出"会**系统性漏掉残差连接和分类头**。

实测（MobileNetV2，FP32 vs QDQ INT8）：

| 选层方式 | 得到层数 | 漏掉了什么 |
|---|---|---|
| 只取 `Conv` 输出名 | 17（+图输出 = 18） | **10 个 `Add`（残差）+ 1 个 `GlobalAveragePool` + 1 个 `Gemm`（分类头）** |
| 全算子候选 ∩ 张量名交集 | **29** | 无 |

而 `Add` 占了可比层的三分之一（10/29）——在 per-channel 的最差层列表里，
`/features/features.13/Add_output_0` 就排在前列。**只取 Conv 就永远看不到它们。**

> 注意：后来核对全量数据发现 **Add 和普通 Conv 的平均 cosine 几乎相同**
> （0.9880 vs 0.9879）。所以 Add 出现在最差榜前列是因为**数量多且位置靠后**，
> 不是因为算子本身更脆弱。准确的说法是"漏掉它们就丢了三分之一的观测层"。

**一个被实验推翻的错误假设，值得完整记录**：

一开始我以为问题更"高级"——ONNX Runtime 的 QDQ 量化器会吞掉 `Clip` 节点并把
输出名复用给生产者 `Conv`，所以我怀疑那 17 个同名张量在两张图里指向**不同的计算**
（FP32 是激活前、INT8 是激活后），数值因此无意义。

验证方法：用两种选层方式跑同一份数据，比较**重叠张量**的数值。

结果：**重叠张量的 cosine 完全相同**（都是 0.980080）。
数值相同 ⟹ 语义相同 ⟹ **我的假设被证伪**。真实问题只是筛选条件太窄。

教训：**越是听起来专业的解释，越容易让你停止验证。**
调试时第一反应往深处想（"一定是算子融合导致语义错位"），
真相却在浅处（"我的筛选范围不够"）。

对应代码：`common.py::intersect_tensors()` / `instrument_explicit()`，
验证脚本：`smoke_test_names.py`，动手复现：`LAB.md` 第 4.3 节。

---

## 8. 注意事项

- `02b_fetch_github_images.py` 首次运行会调 GitHub API 拿文件列表，之后走本地缓存
  （`.cache/imagenet_sample_filelist.json`）。
- `--force` 才会重跑已有产物；默认跳过已存在的模型，节省时间。
- 量化是**串行**的（每个变体都要跑校准），1000 张校准图会很慢，默认用 500 张。
- 导出的 ONNX 是**固定 batch=1**。脚本会用 `common.effective_batch()` 自动把请求的
  batch 收敛到图声明的值，否则会报 `INVALID_ARGUMENT: Got: 4 Expected: 1`。
- 沙箱锁死的临时目录删不掉是**正常的**，不影响使用；跑 `cleanup.ps1` 可以尽力清理。
