# W1 一键运行脚本
#
# 用法（在 C:\Users\xzjdtm\Documents\embedded-ai-sprint 下）：
#   powershell -ExecutionPolicy Bypass -File run_w1.ps1
#
# 可选参数：
#   -SkipFetch      跳过图片下载（已下载过时）
#   -SkipReport     不生成报告
#   -Force          重新生成所有已有产物
#   -NumCalib 500   校准图片数

param(
    [switch]$SkipFetch,
    [switch]$SkipReport,
    [switch]$Force,
    [int]$NumCalib = 500,
    [int]$NumEval = 64
)

$ErrorActionPreference = "Stop"
$WS = $PSScriptRoot
$PY = "D:\miniconda3\envs\rknn-yolov8-train\python.exe"

if (-not (Test-Path $PY)) {
    Write-Host "找不到 Python 3.10 环境: $PY" -ForegroundColor Red
    Write-Host "请确认 conda 环境 rknn-yolov8-train 存在" -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = "$WS\.pylibs"
$env:PYTHONUNBUFFERED = "1"

$script:step = 0
$script:failed = @()

function Step {
    param([string]$Name, [string[]]$CmdArgs)
    $script:step++
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ("[{0}] {1}" -f $script:step, $Name) -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $PY @CmdArgs
    $code = $LASTEXITCODE
    $sw.Stop()
    if ($code -ne 0) {
        Write-Host ("  !! 失败 exit={0} ({1:N1}s)" -f $code, $sw.Elapsed.TotalSeconds) -ForegroundColor Red
        $script:failed += $Name
    } else {
        Write-Host ("  OK ({0:N1}s)" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
    }
}

if (-not (Test-Path "$WS\.pylibs\onnxruntime")) {
    Write-Host "缺少 onnxruntime，先安装：" -ForegroundColor Yellow
    Write-Host "  & D:\miniconda3\python.exe tools\fetch_wheel.py onnxruntime --version 1.17.3 --py cp310 --dest .pylibs --index https://pypi.tuna.tsinghua.edu.cn/pypi"
    exit 1
}

if (-not $SkipFetch) {
    Step "下载真实图片 (GitHub raw, 1000 类)" @(
        "src\w1_quant\02b_fetch_github_images.py", "--download", "--max-classes", "1000"
    )
    Step "构建校准集" @(
        "src\w1_quant\02_prepare_calib.py", "--from-dir", "data\imagenet_sample",
        "--max-images", "$NumCalib"
    )
}

Step "D2 导出 ONNX + onnxsim" @(
    "src\w1_quant\01_export_onnx.py", "--model", "mobilenet_v2", "--opset", "13"
)

$quantArgs = @("src\w1_quant\03_quantize.py")
if ($Force) { $quantArgs += "--force" }
Step "D5 PTQ 量化变体矩阵" $quantArgs

Step "冒烟测试: 张量名对齐" @("src\w1_quant\smoke_test_names.py")

Step "D6-1 逐层误差分析" @(
    "src\w1_quant\04_layer_analysis.py", "--num-images", "32"
)

Step "D6 精度对比" @(
    "src\w1_quant\05_accuracy.py", "--num-images", "200"
)

$sweepArgs = @("src\w1_quant\06_calib_sweep.py", "--num-eval-images", "$NumEval")
if ($Force) { $sweepArgs += "--force" }
Step "D6-2/3 校准集规模与多样性" $sweepArgs

if (-not $SkipReport) {
    Step "D7 生成图表与报告" @("src\w1_quant\07_make_report.py")
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
if ($script:failed.Count -eq 0) {
    Write-Host "全部完成" -ForegroundColor Green
} else {
    Write-Host ("以下步骤失败: " + ($script:failed -join ", ")) -ForegroundColor Red
}
Write-Host ("=" * 70) -ForegroundColor Cyan

Write-Host ""
Write-Host "产出物:"
foreach ($p in @("W1_报告\README.md", "W1_报告\table1_summary.md",
                 "W1_报告\fig1_layer_curve.png", "W1_报告\fig2_calib_sweep.png")) {
    $full = Join-Path $WS $p
    if (Test-Path $full) {
        Write-Host ("  [x] {0}" -f $p) -ForegroundColor Green
    } else {
        Write-Host ("  [ ] {0}  (缺失)" -f $p) -ForegroundColor DarkGray
    }
}
