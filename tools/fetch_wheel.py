"""Download and unpack a wheel into a workspace directory without using pip.

Why this exists
---------------
The DSH file sandbox injects an unoverrideable TEMP directory into every child
process. pip therefore unpacks wheels inside that sandbox temp dir, where the
resulting files cannot be read back, so `pip install --target <workspace>`
fails with `[Errno 13] Permission denied` on the downloaded wheel.

This script avoids pip entirely: it asks the PyPI JSON API for a wheel URL,
downloads the wheel straight into the workspace, and extracts it with
zipfile. The result is a plain directory that can be added to PYTHONPATH.

Usage
-----
    python fetch_wheel.py <package> [--version X.Y.Z] [--dest DIR]
                          [--index https://pypi.tuna.tsinghua.edu.cn/simple]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import sysconfig
import urllib.error
import urllib.request
import zipfile


def wheel_tags(py_override: str | None = None) -> tuple[list[str], str]:
    """Return acceptable platform tags, most specific first."""
    if py_override:
        tags = [py_override]
        major, minor = int(py_override[2]), int(py_override[3:])
        for m in range(minor - 1, max(minor - 3, 5), -1):
            tags.append("cp%d%d" % (major, m))
    else:
        py_tag = "cp%d%d" % sys.version_info[:2]
        tags = [py_tag]
        for minor in range(
            sys.version_info[1] - 1, max(sys.version_info[1] - 3, 5), -1
        ):
            tags.append("cp%d%d" % (sys.version_info[0], minor))

    machine = platform.machine().lower()
    if os.name == "nt":
        arch = "win_amd64" if machine in ("amd64", "x86_64") else "win32"
    else:
        arch = "manylinux2014_x86_64" if machine in ("amd64", "x86_64") else machine
    return tags, arch


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pick_wheel(files: list[dict], tags: list[str], arch: str) -> dict | None:
    candidates = []
    for f in files:
        name = f.get("filename", "")
        if not name.endswith(".whl"):
            continue
        if "py3-none-any" in name:
            candidates.append((0, name, f))  # pure python, always fine
            continue
        for rank, tag in enumerate(tags):
            if tag in name and arch in name:
                candidates.append((rank + 1, name, f))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    best_rank = candidates[0][0]
    for rank, _name, f in candidates:
        if rank == best_rank:
            return f
    return None


def download_with_retry(picked: dict, wheel_path: str, attempts: int = 4) -> None:
    """Download a wheel, resuming and retrying because mirrors often reset."""
    expected = picked.get("size") or 0
    urls = [picked["url"]]
    # A mirror URL cannot be rewritten into a pypi.org URL (different hash path),
    # so ask PyPI for this exact file as a genuine fallback.
    if "pythonhosted.org" not in picked["url"]:
        try:
            meta = fetch_json(f"https://pypi.org/pypi/{picked['filename'].split('-')[0]}/json")
            for candidate in meta.get("urls", []):
                if candidate.get("filename") == picked["filename"]:
                    urls.append(candidate["url"])
                    break
        except Exception as exc:  # noqa: BLE001
            print("(pypi fallback lookup failed: %s)" % type(exc).__name__)

    part = wheel_path + ".part"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        for url in urls:
            offset = os.path.getsize(part) if os.path.exists(part) else 0
            if expected and offset >= expected:
                break
            req = urllib.request.Request(url)
            if offset:
                req.add_header("Range", f"bytes={offset}-")
            print(
                "downloading [%d/%d] offset=%d ..." % (attempt, attempts, offset),
                end=" ",
                flush=True,
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp, open(
                    part, "ab"
                ) as out:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                got = os.path.getsize(part)
                if not expected or got >= expected:
                    print("done (%.1f MB)" % (got / 1e6))
                    os.replace(part, wheel_path)
                    return
                print("partial %.1f/%.1f MB" % (got / 1e6, expected / 1e6))
            except Exception as exc:  # noqa: BLE001 - report and retry any network error
                last_error = exc
                got = os.path.getsize(part) if os.path.exists(part) else 0
                print("reset after %.1f MB (%s)" % (got / 1e6, type(exc).__name__))

    raise RuntimeError("download failed after retries: %r" % (last_error,))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--version", default=None)
    ap.add_argument("--dest", default=".pylibs")
    ap.add_argument("--index", default="https://pypi.org/pypi")
    ap.add_argument("--cache", default=".wheelcache")
    ap.add_argument(
        "--py",
        default=None,
        help="target CPython tag to match, e.g. cp310 (defaults to this interpreter)",
    )
    args = ap.parse_args()

    tags, arch = wheel_tags(args.py)
    print("python   :", platform.python_version(), "|", sys.executable)
    print("tags     :", tags, "| arch:", arch)

    version = None
    files: list[dict] = []

    if args.version:
        # Fetch the whole package index and locate the requested release. This
        # avoids relying on a mirror exposing per-version JSON endpoints.
        all_meta = fetch_json(f"{args.index}/{args.package}/json")
        if args.version in all_meta.get("releases", {}):
            files = all_meta["releases"][args.version]
            version = args.version
        else:
            print("version %s not found; available: %s" % (
                args.version,
                sorted(all_meta.get("releases", {}))[-8:],
            ))
            return 1
        picked = pick_wheel(files, tags, arch)
        if picked is None:
            print("no compatible wheel for version", args.version)
            print("available:", [f.get("filename") for f in files][:20])
            return 1
    else:
        meta = fetch_json(f"{args.index}/{args.package}/json")
        version = meta["info"]["version"]
        files = meta.get("urls") or []
        if not files:
            for ver in sorted(meta["releases"], reverse=True):
                candidate = pick_wheel(meta["releases"][ver], tags, arch)
                if candidate:
                    files, version = meta["releases"][ver], ver
                    break
        picked = pick_wheel(files, tags, arch)
        if picked is None:
            print("no compatible wheel found")
            print("available:", [f.get("filename") for f in files][:20])
            return 1

    print("selected :", picked["filename"], "(version %s)" % version)

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.dest, exist_ok=True)
    wheel_path = os.path.join(args.cache, picked["filename"])

    if not os.path.exists(wheel_path):
        download_with_retry(picked, wheel_path)
    else:
        print("cached   :", wheel_path)

    size_mb = os.path.getsize(wheel_path) / 1e6
    print("wheel    : %.1f MB" % size_mb)

    print("extracting to", os.path.abspath(args.dest), "...", end=" ", flush=True)
    with zipfile.ZipFile(wheel_path) as zf:
        zf.extractall(args.dest)
    print("done")

    print("\nOK. Use it with:")
    print("  set PYTHONPATH=%s" % os.path.abspath(args.dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
