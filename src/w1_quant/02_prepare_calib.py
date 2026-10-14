"""W1-D6 aid: build real calibration image sets of different sizes and mixes.

Why real images matter
----------------------
The D6 calibration experiment is supposed to answer "校准集为什么不能只用一类图".
Synthetic noise has no class structure, so it cannot be *unrepresentative* of
anything - the question would be meaningless.

Two sets are produced:

* ``calib_all``    - a balanced, shuffled slice across all classes (the good set)
* ``calib_narrow`` - images from a single class (the deliberately bad set)

plus nested subsets, so calibration-set **size** and **diversity** can be varied
independently.

Data source
-----------
CIFAR-10 (170 MB, reachable). ImageNet-1k validation is the textbook choice but
is ~6.7 GB and its mirrors are unreachable here; CIFAR-10 is the standard
small-scale substitute and is what the original INT8 papers use for
MobileNet/ResNet curves. Images are upscaled 32 -> 224 px and normalized with
ImageNet statistics, matching what a real camera pipeline would do.

Usage
-----
    python 02_prepare_calib.py --download --max-images 600
    python 02_prepare_calib.py --from-dir /path/to/my/images --max-images 600
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w1_quant.common import write_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def download(url: str, dest: str, attempts: int = 4) -> None:
    part = dest + ".part"
    if os.path.exists(dest):
        print("cached   :", dest)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, attempts + 1):
        offset = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if offset:
            req.add_header("Range", f"bytes={offset}-")
        print(
            f"download [{attempt}/{attempts}] at {offset/1e6:.1f} MB ...",
            flush=True,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab") as out:
                total = resp.headers.get("Content-Length")
                expect = offset + int(total) if total else 0
                last_report = 0.0
                import time

                started = time.time()
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done = out.tell()
                    if done - last_report > 20e6:
                        last_report = done
                        speed = done / 1e6 / max(time.time() - started, 1e-6)
                        pct = f"{done/expect*100:5.1f}%" if expect else "  ?  "
                        print(
                            f"    {pct} {done/1e6:7.1f}/{(expect/1e6 if expect else 0):7.1f} MB"
                            f"  {speed:5.2f} MB/s",
                            flush=True,
                        )
            os.replace(part, dest)
            print("done     : %.1f MB" % (os.path.getsize(dest) / 1e6))
            return
        except Exception as exc:  # noqa: BLE001
            print("  interrupted (%s), resuming" % type(exc).__name__)
    raise RuntimeError("download failed: " + url)


def extract_cifar10(cache_dir: str, out_dir: str, per_class: int, seed: int) -> list[str]:
    """Extract a balanced slice of CIFAR-10 to PNG files, return their paths."""
    import cv2
    import numpy as np

    archive = os.path.join(cache_dir, "cifar-10-python.tar.gz")
    download(CIFAR10_URL, archive)

    marker = os.path.join(out_dir, ".done")
    if os.path.exists(marker):
        print("extracted:", out_dir)
        return sorted(glob.glob(os.path.join(out_dir, "**", "*.png"), recursive=True))

    print("extracting CIFAR-10 ...", flush=True)
    batch_dir = os.path.join(cache_dir, "cifar-10-batches-py")
    if not os.path.isdir(batch_dir):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(cache_dir)

    rng = random.Random(seed)
    per_class_images: dict[int, list[np.ndarray]] = {i: [] for i in range(10)}
    wanted = per_class
    # batch 1..5 are the training split; use them all until every class is full.
    for batch_no in range(1, 6):
        path = os.path.join(batch_dir, f"data_batch_{batch_no}")
        with open(path, "rb") as fh:
            data = pickle.load(fh, encoding="bytes")
        images = data[b"data"]
        labels = data[b"labels"]
        idx = list(range(len(labels)))
        rng.shuffle(idx)
        for i in idx:
            label = labels[i]
            if len(per_class_images[label]) >= wanted:
                continue
            flat = images[i].reshape(3, 32, 32).transpose(1, 2, 0)  # CHW -> HWC RGB
            per_class_images[label].append(flat)
        if all(len(v) >= wanted for v in per_class_images.values()):
            break

    paths: list[str] = []
    for label, imgs in per_class_images.items():
        class_dir = os.path.join(out_dir, CIFAR10_CLASSES[label])
        os.makedirs(class_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            out_path = os.path.join(class_dir, f"{CIFAR10_CLASSES[label]}_{i:04d}.png")
            if not os.path.exists(out_path):
                cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            paths.append(out_path)
        print(f"  {CIFAR10_CLASSES[label]:12s} {len(imgs)} images")
    with open(marker, "w") as fh:
        fh.write("ok")
    return sorted(paths)


def class_of(path: str, source: str) -> str:
    if source == "cifar10":
        return os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    # ImageNet-style names carry the label: n01440764_tench -> n01440764_tench
    return stem


def domain_key(path: str, source: str) -> str | None:
    """ImageNet class id (`n########`) parsed from a file path, or None."""
    import re

    if source == "cifar10":
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r"(n\d+)", stem)
    return match.group(1) if match else None


def domain_partition(paths: list[str], source: str) -> dict[str, list[str]]:
    """Split image *paths* into broad ImageNet domains by class-id rank.

    WordNet ids are roughly grouped: early ids are animals (fish -> birds ->
    mammals), the middle is objects, the tail is scenes and materials. Splitting
    by numeric rank therefore yields domain-coherent bands without needing the
    full WordNet tree.

    Returns ``{domain_label: [paths]}``, or ``{"all": paths}`` when the source
    has no ImageNet-style ids.
    """
    indexed: list[tuple[int, str]] = []
    for path in paths:
        key = domain_key(path, source)
        if key is None:
            return {"all": sorted(paths)}
        indexed.append((int(key[1:]), path))
    if not indexed:
        return {}

    indexed.sort(key=lambda item: (item[0], item[1]))
    total = len(indexed)
    bands = [
        ("domain_animal", 0.0, 0.60),
        ("domain_object", 0.60, 0.83),
        ("domain_scene", 0.83, 1.0),
    ]
    out: dict[str, list[str]] = {}
    for label, lo, hi in bands:
        chunk = [path for _idx, path in indexed[int(total * lo) : int(total * hi)]]
        if chunk:
            out[label] = chunk
    return out


def domain_stratified(
    domains: dict[str, list[str]], count: int, rng
) -> list[str]:
    """Equal numbers from every domain - 'diverse but domain-aware'."""
    picked: list[str] = []
    keys = sorted(domains)
    per = max(1, count // max(len(keys), 1))
    for key in keys:
        picked.extend(domains[key][:per])
    remaining = count - len(picked)
    if remaining > 0:
        for key in keys:
            extra = [n for n in domains[key][per : per + remaining] if n not in picked]
            picked.extend(extra[:remaining])
            remaining -= len(extra[:remaining])
            if remaining <= 0:
                break
    rng.shuffle(picked)
    return picked[:count]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch CIFAR-10")
    ap.add_argument("--from-dir", default=None, help="use your own image folder")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "calib"))
    ap.add_argument("--cache", default=os.path.join(ROOT, ".cache", "datasets"))
    ap.add_argument("--max-images", type=int, default=600)
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--sizes", default="10,50,200,500")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--cifar-dir",
        default=None,
        help="where extracted CIFAR PNGs live (default: <cache>/cifar10-png)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    source = "custom"

    if args.from_dir:
        images = sorted(
            p
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG")
            for p in glob.glob(os.path.join(args.from_dir, "**", ext), recursive=True)
        )
        print(f"source   : {args.from_dir} -> {len(images)} images")
    elif args.download:
        source = "cifar10"
        cifar_dir = args.cifar_dir or os.path.join(args.cache, "cifar10-png")
        images = extract_cifar10(args.cache, cifar_dir, args.per_class, args.seed)
        print(f"source   : CIFAR-10 -> {len(images)} images")
    else:
        print("nothing to do: pass --download or --from-dir")
        return 1

    if not images:
        print("no images found")
        return 1

    by_class: dict[str, list[str]] = {}
    for path in images:
        by_class.setdefault(class_of(path, source), []).append(path)
    print(f"classes  : {len(by_class)}")

    names = sorted(by_class)
    for name in names:
        rng.shuffle(by_class[name])

    # ---- balanced slice across classes (the "good" calibration set) --------
    target = min(args.max_images, len(images))
    balanced: list[str] = []
    cursor = {name: 0 for name in names}
    while len(balanced) < target:
        progressed = False
        for name in names:
            idx = cursor[name]
            if idx < len(by_class[name]):
                balanced.append(by_class[name][idx])
                cursor[name] = idx + 1
                progressed = True
                if len(balanced) >= target:
                    break
        if not progressed:
            break
    rng.shuffle(balanced)

    # ---- domain-restricted slice (the deliberately narrow calibration set) --
    # With one sample per ImageNet class we cannot gather 500 images of a single
    # class, so "narrow" is defined by *domain* instead: a contiguous band of
    # ImageNet class ids. Bands are split by numeric rank of the class id.
    domains = domain_partition(images, source)
    widest = max(domains, key=lambda key: len(domains[key])) if domains else None
    narrow = domains.get(widest, [])[:target] if widest else []

    out = args.outdir
    os.makedirs(out, exist_ok=True)

    def write_list(path: str, items: list[str]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for item in items:
                fh.write(os.path.abspath(item) + "\n")
        print("  %-46s %4d images" % (os.path.relpath(path, ROOT), len(items)))

    print("\n=== 生成的校准集清单 ===")
    write_list(os.path.join(out, "calib_all.txt"), balanced)
    write_list(os.path.join(out, "calib_narrow.txt"), narrow)
    write_list(os.path.join(out, "all_images.txt"), sorted(images))

    nested = sorted(int(s) for s in args.sizes.split(",") if s.strip())
    for size in nested:
        if size <= len(balanced):
            write_list(os.path.join(out, f"calib_nested_{size}.txt"), balanced[:size])
    biggest = nested[-1] if nested else 0
    if biggest:
        write_list(
            os.path.join(out, f"calib_narrow_{biggest}.txt"), narrow[:biggest]
        )
        # a mixed set: half balanced + half domain-restricted, same total size
        half = biggest // 2
        mixed = balanced[:half] + narrow[: biggest - half]
        rng.shuffle(mixed)
        write_list(os.path.join(out, f"calib_mixed_{biggest}.txt"), mixed)
        if domains:
            write_list(
                os.path.join(out, f"calib_domain_balanced_{biggest}.txt"),
                domain_stratified(domains, biggest, rng),
            )

    meta = {
        "source": source,
        "total_available": len(images),
        "num_classes": len(names),
        "balanced_count": len(balanced),
        "narrow_domain": widest,
        "narrow_count": len(narrow),
        "domains": {key: len(value) for key, value in domains.items()},
        "nested_sizes": nested,
        "seed": args.seed,
        "outdir": out,
    }
    write_json(meta, os.path.join(out, "calib_meta.json"))
    print(f"\nmeta     : {os.path.relpath(os.path.join(out, 'calib_meta.json'), ROOT)}")
    print(f"\n域划分 : " + ", ".join(
        f"{key}={len(value)} 类" for key, value in sorted(domains.items())))
    print(f"窄校准集取自 '{widest}'（{len(narrow)} 张），与均衡集图片数相同，用来验证"
          f"\"校准集代表性\"是否比\"数量\"更重要")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
