import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from exp1.common import EXP
from exp1.downloads import resume_ranges


class Response(io.BytesIO):
    def __init__(self, data, start, end, total, fail_after=None):
        super().__init__(data[start:end + 1])
        self.status = 206
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}
        self.fail_after = fail_after

    def read(self, n):
        if self.fail_after is not None:
            if self.tell() >= self.fail_after:
                raise TimeoutError("injected transient read timeout")
            n = min(n, self.fail_after - self.tell())
        return super().read(n)


class RangeDownloadTests(unittest.TestCase):
    def test_partial_write_timeout_resumes_without_repeating_bytes(self):
        data = b"0123456789abcdefghijklmnopqrstuvwxyz"
        calls = []
        def open_range(request, timeout):
            start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
            calls.append((start, end))
            return Response(data, start, end, len(data), fail_after=3 if len(calls) == 1 else None)
        with tempfile.TemporaryDirectory(dir=EXP / "tests") as directory:
            path = Path(directory) / "weights.partial"
            path.write_bytes(data[:5])
            with patch("exp1.downloads.urllib.request.urlopen", side_effect=open_range), patch("exp1.downloads.time.sleep"):
                resume_ranges("https://example.invalid/weights", path, len(data), range_bytes=8)
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(calls[:2], [(5, 12), (8, 15)])

    def test_invalid_content_range_is_not_retried_or_appended(self):
        data = b"0123456789"
        response = Response(data, 0, 7, len(data))
        with tempfile.TemporaryDirectory(dir=EXP / "tests") as directory:
            path = Path(directory) / "weights.partial"
            path.write_bytes(data[:3])
            with patch("exp1.downloads.urllib.request.urlopen", return_value=response) as opener:
                with self.assertRaisesRegex(ValueError, "did not honor Range"):
                    resume_ranges("https://example.invalid/weights", path, len(data), range_bytes=8)
            self.assertEqual(path.read_bytes(), data[:3])
            self.assertEqual(opener.call_count, 1)

    def test_retry_budget_is_bounded_and_missing_file_remains_missing(self):
        with tempfile.TemporaryDirectory(dir=EXP / "tests") as directory:
            path = Path(directory) / "weights.partial"
            with patch("exp1.downloads.urllib.request.urlopen", side_effect=TimeoutError("injected")) as opener, patch("exp1.downloads.time.sleep"):
                with self.assertRaises(TimeoutError):
                    resume_ranges("https://example.invalid/weights", path, 10, max_retries=2)
            self.assertEqual(opener.call_count, 3)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
