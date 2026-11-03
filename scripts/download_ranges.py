#!/usr/bin/env python3
"""Download a large HTTP file with bounded parallel byte-range requests."""

from __future__ import annotations

import argparse
import errno
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--connections", type=int, default=12)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for each socket read before retrying a range.",
    )
    parser.add_argument(
        "--size-bytes",
        type=int,
        help="Known file size; avoids a separate HEAD request.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep the existing contiguous prefix and download only the tail.",
    )
    parser.add_argument(
        "--resume-sparse",
        action="store_true",
        help=(
            "For a preallocated file, keep allocated data and redownload every "
            "chunk that still contains a filesystem hole."
        ),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Report sparse missing chunks without modifying or downloading data.",
    )
    return parser.parse_args()


def resolve_size(url: str) -> int:
    size = None
    for attempt in range(1, 6):
        try:
            response = requests.head(url, allow_redirects=True, timeout=(30, 60))
            response.raise_for_status()
            size = int(response.headers["Content-Length"])
            if size > 1:
                break
        except (requests.RequestException, KeyError, ValueError):
            if attempt == 5:
                break
            time.sleep(2**attempt)

    if size is None or size <= 1:
        response = requests.get(
            url,
            headers={"Range": "bytes=0-0"},
            allow_redirects=True,
            timeout=(30, 60),
        )
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes\s+\d+-\d+/(\d+)", content_range)
        if not match:
            raise RuntimeError(
                "server did not provide a usable Content-Length or Content-Range"
            )
        size = int(match.group(1))
    if size <= 0:
        raise RuntimeError("server returned an invalid Content-Length")
    return size


def fetch_range(
    url: str,
    output: Path,
    start: int,
    end: int,
    retries: int,
    read_timeout: float,
) -> int:
    expected = end - start + 1
    for attempt in range(1, retries + 1):
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            with requests.get(
                url,
                headers=headers,
                allow_redirects=True,
                stream=True,
                timeout=(30, read_timeout),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"range request returned HTTP {response.status_code}"
                    )
                received = 0
                with output.open("r+b", buffering=0) as handle:
                    handle.seek(start)
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        handle.write(block)
                        received += len(block)
                if received != expected:
                    raise RuntimeError(
                        f"range {start}-{end}: expected {expected}, got {received}"
                    )
                return received
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int


def allocated_ranges(path: Path, size: int) -> list[ByteRange]:
    """Return extents backed by data using the filesystem's SEEK_DATA API."""
    ranges: list[ByteRange] = []
    descriptor = os.open(path, os.O_RDONLY)
    try:
        offset = 0
        while offset < size:
            try:
                data_start = os.lseek(descriptor, offset, os.SEEK_DATA)
            except OSError as exc:
                if exc.errno == errno.ENXIO:
                    break
                if exc.errno in (errno.EINVAL, errno.ENOTSUP):
                    raise RuntimeError(
                        "the filesystem does not support SEEK_DATA/SEEK_HOLE"
                    ) from exc
                raise
            data_end = min(
                os.lseek(descriptor, data_start, os.SEEK_HOLE),
                size,
            )
            ranges.append(ByteRange(data_start, data_end))
            offset = data_end
    finally:
        os.close(descriptor)
    return ranges


def missing_chunks(
    path: Path,
    size: int,
    chunk_size: int,
) -> list[tuple[int, int]]:
    """Find chunks that are not completely backed by allocated file data."""
    allocated = allocated_ranges(path, size)
    missing: list[tuple[int, int]] = []
    data_index = 0
    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)
        while (
            data_index < len(allocated)
            and allocated[data_index].end <= start
        ):
            data_index += 1
        cursor = start
        covered = True
        probe = data_index
        while cursor < end and probe < len(allocated):
            extent = allocated[probe]
            if extent.start > cursor:
                covered = False
                break
            cursor = max(cursor, extent.end)
            probe += 1
        if cursor < end:
            covered = False
        if not covered:
            missing.append((start, end - 1))
    return missing


def main() -> None:
    args = parse_args()
    if args.resume and args.resume_sparse:
        raise ValueError("--resume and --resume-sparse are mutually exclusive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    size = args.size_bytes if args.size_bytes is not None else resolve_size(args.url)
    if size <= 0:
        raise RuntimeError("file size must be positive")

    chunk_size = args.chunk_mib * 1024 * 1024
    start_offset = 0
    if args.resume and output.is_file():
        start_offset = output.stat().st_size
        if start_offset > size:
            raise RuntimeError(
                f"existing file is larger than expected: {start_offset} > {size}"
            )

    sparse_ranges = None
    if args.resume_sparse:
        if not output.is_file():
            raise FileNotFoundError(
                f"--resume-sparse requires an existing file: {output}"
            )
        if output.stat().st_size != size:
            raise RuntimeError(
                f"sparse resume requires size {size}, got {output.stat().st_size}"
            )
        sparse_ranges = missing_chunks(output, size, chunk_size)
        recovered = sum(end - start + 1 for start, end in sparse_ranges)
        print(
            f"sparse resume: {len(sparse_ranges)} incomplete chunks, "
            f"{recovered} bytes to download"
        )
        if args.report_only:
            return

    output.touch(exist_ok=True)
    with output.open("r+b") as handle:
        handle.truncate(size)

    ranges = (
        sparse_ranges
        if sparse_ranges is not None
        else [
            (start, min(start + chunk_size - 1, size - 1))
            for start in range(start_offset, size, chunk_size)
        ]
    )
    completed = size - sum(end - start + 1 for start, end in ranges)
    with tqdm(
        total=size,
        initial=start_offset,
        unit="B",
        unit_scale=True,
        desc=output.name,
    ) as progress:
        with ThreadPoolExecutor(max_workers=args.connections) as executor:
            futures = [
                executor.submit(
                    fetch_range,
                    args.url,
                    output,
                    start,
                    end,
                    args.retries,
                    args.read_timeout,
                )
                for start, end in ranges
            ]
            for future in as_completed(futures):
                completed += future.result()
                progress.update(completed - progress.n)

    actual = os.path.getsize(output)
    if actual != size:
        raise RuntimeError(f"size mismatch: expected {size}, got {actual}")
    if args.resume_sparse:
        remaining = missing_chunks(output, size, chunk_size)
        if remaining:
            raise RuntimeError(
                f"{len(remaining)} incomplete chunks remain after sparse resume"
            )
    print(f"verified {output}: {actual} bytes")


if __name__ == "__main__":
    main()
