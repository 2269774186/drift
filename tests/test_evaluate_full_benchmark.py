from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_full_benchmark import (
    YUV420Video,
    discover_videos,
    parse_fps,
)


class YUV420VideoTest(unittest.TestCase):
    def test_reads_complete_i420_frame_and_converts_to_bgr(self) -> None:
        width = 4
        height = 4
        frame_bytes = width * height * 3 // 2
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.yuv"
            data = np.zeros((2, frame_bytes), dtype=np.uint8)
            data[:, : width * height] = 126
            data[:, width * height :] = 128
            path.write_bytes(data.tobytes())

            video = YUV420Video(path, width, height, 2, "limited")

            self.assertEqual(video.shape, (2, height, width, 3))
            self.assertEqual(len(video), 2)
            self.assertEqual(video[0].shape, (height, width, 3))
            self.assertEqual(video[0].dtype, np.uint8)
            self.assertTrue(np.all(np.abs(video[0].astype(int) - 128) <= 2))

    def test_rejects_wrong_file_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yuv"
            path.write_bytes(b"\0" * 10)
            with self.assertRaises(ValueError):
                YUV420Video(path, 4, 4, 1, "limited")


class BenchmarkDiscoveryTest(unittest.TestCase):
    def test_parses_encoded_video_fps(self) -> None:
        self.assertEqual(
            parse_fps(Path("Beauty_1920_1080_120_96f.yuv")),
            120.0,
        )
        self.assertEqual(
            parse_fps(Path("videoSRC01_1920_1080_30fps_96f.yuv")),
            30.0,
        )
        self.assertEqual(parse_fps(Path("unknown.yuv")), 30.0)

    def test_discovers_and_limits_grouped_yuv_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "uvg").mkdir()
            (root / "hevc_b").mkdir()
            (root / "uvg/b.yuv").touch()
            (root / "uvg/a.yuv").touch()
            (root / "hevc_b/c.yuv").touch()
            (root / "hevc_b/ignored.txt").touch()

            entries = discover_videos(root, ("uvg", "hevc_b"), 0)
            limited = discover_videos(root, ("uvg", "hevc_b"), 2)

            self.assertEqual(
                [entry["name"] for entry in entries],
                ["a", "b", "c"],
            )
            self.assertEqual(
                [entry["name"] for entry in limited],
                ["a", "b"],
            )


if __name__ == "__main__":
    unittest.main()
