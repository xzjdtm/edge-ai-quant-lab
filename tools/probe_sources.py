"""Measure which download sources are actually usable from this machine.

Why this exists
---------------
"Just download the dataset" is not a workable instruction everywhere. On this
machine the official dataset hosts are throttled to ~0.02 MB/s while a couple of
mirrors are 50-80x faster, and one popular mirror serves a truncated file that
looks like success until you try to open it. Guessing wastes hours, so measure.

Measured on 2026-10-13 (WAN, no proxy):

    ==========================================  ==========  ==========
    source                                      throughput  usable
    ==========================================  ==========  ==========
    pypi.tuna.tsinghua.edu.cn (PyPI)            ~1.6 MB/s   yes
    hf-mirror.com                               ~1.2 MB/s   yes
    raw.githubusercontent.com                   ~0.33 MB/s  yes
    mirror.sjtu.edu.cn  /cifar/                 truncated   NO (fake)
    cs.toronto.edu (CIFAR-10 official)          ~0.02 MB/s  no
    robots.ox.ac.uk (Oxford Pets)               ~0.03 MB/s  no
    github.com releases                         timeout     no
    ==========================================  ==========  ==========

The practical consequence is recorded in ENVIRONMENT.md: images come from
`raw.githubusercontent.com` one file at a time, because a reset then costs a
single ~100 KB image instead of an entire archive.

Usage
-----
    python probe_sources.py                 # full sweep
    python probe_sources.py --pypi-only      # just the package mirrors
    python probe_sources.py --bytes 8000000  # download more per source
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

#: PyPI mirrors, checked because wheels are the fastest reliable bulk transfer.
PYPI_MIRRORS = [
    ("tuna", "https://pypi.tuna.tsinghua.edu.cn/simple/onnxruntime/"),
    ("aliyun", "https://mirrors.aliyun.com/pypi/simple/onnxruntime/"),
    ("ustc", "https://mirrors.ustc.edu.cn/pypi/simple/onnxruntime/"),
    ("official pypi", "https://pypi.org/simple/onnxruntime/"),
]

#: Dataset hosts and mirrors, for real images.
IMAGE_SOURCES = [
    ("hf-mirror: cifar10 parquet",
     "https://hf-mirror.com/datasets/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet"),
    ("github raw: imagenet sample",
     "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01440764_tench.JPEG"),
    ("mirror.sjtu: cifar tarball",
     "https://mirror.sjtu.edu.cn/cifar/cifar-10-python.tar.gz"),
    ("cs.toronto: cifar tarball",
     "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"),
    ("robots.ox: oxford pets",
     "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"),
    ("github releases: yolov8n",
     "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"),
]

#: PyTorch's own weight host, needed even if every dataset mirror is dead.
WEIGHT_SOURCES = [
    ("download.pytorch.org: mobilenet_v2",
     "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"),
]


def measure(name: str, url: str, budget_bytes: int, window: float) -> float:
    """Download up to `budget_bytes` within `window` seconds; return MB/s."""
    started = time.time()
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as resp:
            declared = resp.headers.get("Content-Length")
            got = 0
            t0 = time.time()
            while got < budget_bytes and time.time() - t0 < window:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                got += len(chunk)
            elapsed = max(time.time() - t0, 1e-6)
            rate = got / 1e6 / elapsed

            if got < budget_bytes and declared and got < int(declared):
                # Connection ended early: suspicious, this is the "fake mirror"
                # signature where a small prefix is served and then closed.
                verdict = "TRUNCATED"
            elif rate > 0.3:
                verdict = "FAST"
            elif rate > 0.1:
                verdict = "ok"
            else:
                verdict = "slow"

            print("  %-9s %-30s %7.2f MB in %5.1fs = %6.3f MB/s  (http %s%s)"
                  % (verdict, name[:30], got / 1e6, elapsed, rate, resp.status,
                     ", declared %.1f MB" % (int(declared) / 1e6) if declared else ""))
            return rate
    except urllib.error.HTTPError as exc:
        print("  HTTP%-5d %-30s %s" % (exc.code, name[:30], exc.reason))
    except Exception as exc:  # noqa: BLE001
        print("  FAIL     %-30s %s (%.1fs)"
              % (name[:30], type(exc).__name__, time.time() - started))
    return 0.0


def section(title: str, entries: list[tuple[str, str]], budget: int, window: float) -> dict:
    print("=" * 78)
    print(title)
    print("=" * 78)
    results = {}
    for name, url in entries:
        results[name] = measure(name, url, budget, window)
    print()
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pypi-only", action="store_true")
    ap.add_argument("--bytes", type=int, default=4_000_000,
                    help="bytes to pull per source before moving on")
    ap.add_argument("--window", type=float, default=15.0,
                    help="seconds to allow per source")
    args = ap.parse_args()

    results: dict = {}
    results.update(section("PyPI mirrors (wheels)", PYPI_MIRRORS,
                           min(args.bytes, 1_000_000), args.window))
    if not args.pypi_only:
        results.update(section("Image sources", IMAGE_SOURCES,
                               args.bytes, args.window))
        results.update(section("Model weights", WEIGHT_SOURCES,
                               min(args.bytes, 2_000_000), args.window))

    good = [(n, r) for n, r in results.items() if r > 0.3]
    print("=" * 78)
    if good:
        print("可用源（>0.3 MB/s），应优先使用：")
        for name, rate in sorted(good, key=lambda kv: -kv[1]):
            print("  %6.2f MB/s  %s" % (rate, name))
    else:
        print("没有找到快速源；需要走逐文件小请求策略（见 02b_fetch_github_images.py）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
