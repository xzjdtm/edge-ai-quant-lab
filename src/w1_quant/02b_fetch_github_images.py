"""Fetch real photographs: one ImageNet image per class, via GitHub raw.

Why this source
---------------
Measured throughput on this machine:

* cs.toronto.edu / robots.ox.ac.uk / github releases  ~0.02 MB/s  (unusable)
* mirror.sjtu.edu.cn "cifar" path                     5 MB then truncated (fake)
* hf-mirror.com                                       ~1.2 MB/s but the parquet
  copy needs pyarrow, which stalls partway
* **raw.githubusercontent.com**                        ~0.33 MB/s, small files

`EliSchwartz/imagenet-sample-images` holds one representative photograph for
each of the 1000 ImageNet classes, ~100 KB each. Small files also mean a reset
costs one image instead of a whole archive, which matters on this network.

The result is genuinely diverse real imagery with class labels, which is exactly
what the calibration-set diversity experiment needs.

Usage
-----
    python 02b_fetch_github_images.py --list
    python 02b_fetch_github_images.py --download --max-per-class 1 --max-classes 1000
    python 02b_fetch_github_images.py --download --max-per-class 3   # more images
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "EliSchwartz/imagenet-sample-images"
API = f"https://api.github.com/repos/{REPO}/contents/"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/"
UA = {"User-Agent": "Mozilla/5.0"}


def list_repo_files(cache_path: str, refresh: bool = False) -> list[str]:
    """Return every image filename in the repo (cached to disk)."""
    if os.path.exists(cache_path) and not refresh:
        with open(cache_path, encoding="utf-8") as fh:
            return json.load(fh)

    names: list[str] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{REPO}/git/trees/master?recursive=1"
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            print("git trees API failed (%s); falling back to contents API" % exc)
            req = urllib.request.Request(API, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = {"tree": json.loads(resp.read().decode("utf-8"))}
        for entry in data.get("tree", []):
            path = entry.get("path", "")
            if path.lower().endswith((".jpeg", ".jpg", ".png")):
                names.append(path)
        break

    names = sorted(set(names))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(names, fh, indent=1)
    return names


def class_of(filename: str) -> str:
    """`n01440764_tench.JPEG` -> `n01440764`."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split("_")[0]


def download_one(rel_path: str, dest_root: str, attempts: int = 3) -> tuple[str, int]:
    dest = os.path.join(dest_root, rel_path)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return dest, os.path.getsize(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(RAW + rel_path, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < 1024:
                raise ValueError("suspiciously small file")
            with open(dest, "wb") as fh:
                fh.write(data)
            return dest, len(data)
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            time.sleep(0.5)
    raise RuntimeError(rel_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--refresh-list", action="store_true")
    ap.add_argument("--max-per-class", type=int, default=1)
    ap.add_argument("--max-classes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "imagenet_sample"))
    ap.add_argument("--cache", default=os.path.join(ROOT, ".cache"))
    args = ap.parse_args()

    if not (args.list or args.download):
        ap.print_help()
        return 1

    cache_path = os.path.join(args.cache, "imagenet_sample_filelist.json")
    files = list_repo_files(cache_path, refresh=args.refresh_list)
    print("repo files : %d" % len(files))

    by_class: dict[str, list[str]] = {}
    for name in files:
        by_class.setdefault(class_of(name), []).append(name)
    classes = sorted(by_class)
    print("classes    : %d" % len(classes))

    if args.list:
        for cls in classes[:5]:
            print("  %s -> %s" % (cls, by_class[cls][0]))
        print("  ...")
        return 0

    selected: list[str] = []
    for cls in classes[: args.max_classes]:
        selected.extend(sorted(by_class[cls])[: args.max_per_class])
    print("to download: %d images" % len(selected))

    ok, failed = 0, 0
    total_bytes = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, rel, args.outdir): rel for rel in selected
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            rel = futures[fut]
            try:
                _path, size = fut.result()
                ok += 1
                total_bytes += size
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if failed <= 5:
                    print("  FAIL %s (%s)" % (rel, type(exc).__name__))
            if i % 50 == 0 or i == len(selected):
                elapsed = max(time.time() - t0, 1e-6)
                print(
                    "  %4d/%d  ok=%d fail=%d  %.2f MB  %.2f img/s  %.2f MB/s"
                    % (i, len(selected), ok, failed, total_bytes / 1e6,
                       i / elapsed, total_bytes / 1e6 / elapsed),
                    flush=True,
                )

    print("\ndownloaded : %d images (%.1f MB), failed %d" % (ok, total_bytes / 1e6, failed))
    print("outdir     : %s" % args.outdir)
    if ok < 50:
        print("\nWARNING: too few images for a meaningful calibration study")
        return 1
    print("\nOK - now run 02_prepare_calib.py --from-dir \"%s\"" % args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
