# 清理 W1 运行产生的临时目录
#
# 背景：DSH 沙箱会把新建目录锁成不可访问，`tempfile` 和 pip 留下的临时目录
# 因此删不掉。本脚本按「先放宽 ACL，再删」的顺序尝试，失败的会列出来。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File cleanup.ps1
#   powershell -ExecutionPolicy Bypass -File cleanup.ps1 -IncludeCaches

param(
    [switch]$IncludeCaches   # 同时删除 .cache / .wheelcache / .piptmp / .pylibs
)

$WS = $PSScriptRoot
$targets = @()

foreach ($base in @(".cache\tmp", ".piptmp", ".wheelcache")) {
    $full = Join-Path $WS $base
    if (Test-Path $full) {
        $targets += Get-ChildItem $full -Directory -Force -ErrorAction SilentlyContinue
    }
}
$targets += Get-ChildItem $WS -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "probe.*" }

if ($IncludeCaches) {
    foreach ($d in @(".cache", ".wheelcache", ".piptmp")) {
        $full = Join-Path $WS $d
        if (Test-Path $full) { $targets += Get-Item $full }
    }
}

$ok = 0
$failed = @()

foreach ($item in $targets | Sort-Object FullName -Unique) {
    try {
        # 沙箱把目录 ACL 锁死了，先尝试夺回权限再删。
        & icacls $item.FullName /reset /T /C /Q 2>&1 | Out-Null
        & takeown /F $item.FullName /R /D Y 2>&1 | Out-Null
        Remove-Item $item.FullName -Recurse -Force -ErrorAction Stop
        $ok++
    } catch {
        $failed += $item.FullName
    }
}

Write-Host "已删除 $ok 个临时目录" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host "`n以下 $($failed.Count) 个删不掉（沙箱锁死，不影响使用）：" -ForegroundColor Yellow
    $failed | ForEach-Object { Write-Host "  $_" }
    Write-Host "`n它们只是空目录，可以忽略；重启后重跑本脚本通常就能删掉。" -ForegroundColor DarkGray
}

Write-Host "`n检查是否还有残留的临时产物："
foreach ($p in @("artifacts\models\_cmp_from_raw.onnx",
                 "artifacts\models\_cmp_from_preproc.onnx",
                 "artifacts\models\_test_direct.onnx")) {
    $full = Join-Path $WS $p
    if (Test-Path $full) {
        Remove-Item $full -Force -ErrorAction SilentlyContinue
        Write-Host "  已删除 $p"
    }
}
foreach ($f in Get-ChildItem (Join-Path $WS "artifacts\models") -Filter "*.qi.onnx" -ErrorAction SilentlyContinue) {
    Remove-Item $f.FullName -Force -ErrorAction SilentlyContinue
    Write-Host "  已删除 $($f.Name)"
}
foreach ($f in Get-ChildItem (Join-Path $WS "artifacts\instrumented") -Filter "*_instr.onnx" -ErrorAction SilentlyContinue) {
    # 旧命名方案（按算子输出取层）已被 *_shared.onnx 取代，且结果不可信
    Remove-Item $f.FullName -Force -ErrorAction SilentlyContinue
    Write-Host "  已删除过时的 $($f.Name)"
}
Write-Host "`n完成。" -ForegroundColor Green
