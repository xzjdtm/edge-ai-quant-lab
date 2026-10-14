"""Download real images from the sources that actually work here.

Network reality on this machine (measured, not assumed):

* cs.toronto.edu, robots.ox.ac.uk  ~0.02 MB/s  -> unusable
* Tsinghua PyPI mirror            ~1.6  MB/s  -> usable
* hf-mirror.com                   ~0.5  MB/s  -> usable
* mirror.sjtu.edu.cn              ~5    MB/s  -> usable, needs redirect handling

So the plan is: try the fast CIFAR-10 filesystem mirror first, and fall back to
the HuggingFace parquet copy (decoded with pyarrow) if that fails. Everything is
resumable, because long single-shot downloads tend to get reset here.

Usage
-----
    python 02a_fetch_images.py --check
    python 02a_fetch_images.py --download
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CIFAR_TARBALL_MIRRORS = [
    "https://mirror.sjtu.edu.cn/cifar/cifar-10-python.tar.gz",
    "https://mirrors.tuna.tsinghua.edu.cn/cifar/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]

CIFAR_PARQUET_SOURCES = [
    "https://hf-mirror.com/datasets/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
    "https://hf-mirror.com/datasets/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
]

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def download(url: str, dest: str, attempts: int = 5, expect_min: int = 0) -> str:
    """Resumable download that tolerates the frequent resets seen here."""
    part = dest + ".part"
    if os.path.exists(dest) and os.path.getsize(dest) > expect_min:
        print("  cached: %.1f MB" % (os.path.getsize(dest) / 1e6))
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    total = 0
    for attempt in range(1, attempts + 1):
        offset = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers=UA)
        if offset:
            req.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                length = resp.headers.get("Content-Length")
                if length and offset and resp.status == 200:
                    # Server ignored the Range header; restart cleanly.
                    print("  server ignored Range, restarting")
                    offset = 0
                total = offset + int(length) if length else 0
                mode = "ab" if offset else "wb"
                t0 = time.time()
                last = 0
                with open(part, mode) as out:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        done = out.tell()
                        if done - last > 25e6 or (total and done >= total):
                            last = done
                            speed = done / 1e6 / max(time.time() - t0, 1e-6)
                            print(
                                "  %.1f/%.1f MB  %.2f MB/s"
                                % (done / 1e6, total / 1e6 if total else 0, speed),
                                flush=True,
                            )
            got = os.path.getsize(part)
            if total and got < total:
                print("  partial %.1f/%.1f MB, resuming" % (got / 1e6, total / 1e6))
                continue
            os.replace(part, dest)
            print("  done: %.1f MB" % (os.path.getsize(dest) / 1e6))
            return dest
        except urllib.error.HTTPError as exc:
            print("  HTTP %d on attempt %d" % (exc.code, attempt))
            if exc.code in (403, 404):
                raise
        except Exception as exc:  # noqa: BLE001
            got = os.path.getsize(part) if os.path.exists(part) else 0
            print("  reset at %.1f MB (%s), attempt %d" % (got / 1e6, type(exc).__name__, attempt))
    raise RuntimeError("download failed: " + url)


def fetch_via_mirror(cache: str) -> str | None:
    dest = os.path.join(cache, "cifar-10-python.tar.gz")
    for url in CIFAR_TARBALL_MIRRORS:
        print("mirror: %s" % url)
        try:
            download(url, dest, expect_min=150_000_000)
            return dest
        except Exception as exc:  # noqa: BLE001
            print("  failed: %s" % type(exc).__name__)
    return None


def fetch_via_parquet(cache: str) -> list[str]:
    out = []
    for i, url in enumerate(CIFAR_PARQUET_SOURCES):
        dest = os.path.join(cache, "cifar10_%d.parquet" % i)
        print("parquet: %s" % url)
        try:
            download(url, dest, expect_min=1_000_000)
            out.append(dest)
        except Exception as exc:  # noqa: BLE001
            print("  failed: %s" % type(exc).__name__)
    return out


def verify_tarball(path: str) -> None:
    with tarfile.open(path, "r:gz") as tf:
        names = tf.getnames()
    print("  contains %d members; e.g. %s" % (len(names), names[1] if len(names) > 1 else "?"))


def verify_parquet(path: str) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  pyarrow missing - cannot verify (run fetch_wheel.py pyarrow)")
        return
    table = pq.read_table(path)
    print("  rows=%d columns=%s" % (table.num_rows, table.column_names))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--cache", default=os.path.join(ROOT, ".cache", "datasets"))
    args = ap.parse_args()

    if not (args.check or args.download):
        ap.print_help()
        return 1

    print("cache: %s\n" % args.cache)

    tarball = fetch_via_mirror(args.cache)
    if tarball:
        print("\nverifying tarball:")
        try:
            verify_tarball(tarball)
            print("\nOK - tarball route works. Run 02_prepare_calib.py")
            return 0
        except Exception as exc:  # noqa: BLE001
            print("  corrupt: %s" % exc)
            os.remove(tarball)

    print("\nfalling back to parquet route")
    parcats = fetch_via_parquet(args.cache)
    if not parcats:
        print("\nno route worked")
        return 1
    for path in parcats:
        verify_parquet(path)
    print("\nOK - parquet route works. Run 02_prepare_calib.py --parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
