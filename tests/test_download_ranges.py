from pathlib import Path
import tempfile
import unittest

from scripts.download_ranges import missing_chunks


class DownloadRangesTest(unittest.TestCase):
    def test_missing_chunks_finds_sparse_file_holes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sparse.bin"
            with path.open("wb") as handle:
                handle.truncate(16 * 1024)
                handle.seek(0)
                handle.write(b"x" * 4096)
                handle.seek(8192)
                handle.write(b"y" * 4096)
            try:
                ranges = missing_chunks(path, path.stat().st_size, 4096)
            except RuntimeError as exc:
                self.skipTest(str(exc))
            self.assertEqual(ranges, [(4096, 8191), (12288, 16383)])


if __name__ == "__main__":
    unittest.main()
