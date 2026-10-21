"""Bounded HTTP Range transfers with visible retries for transient I/O failures."""
import http.client
from pathlib import Path
import time
import urllib.error
import urllib.request


def resume_ranges(url, partial, expected_size, *, range_bytes=64 << 20, max_retries=12, timeout=120):
    """Preserve written bytes; require exact ranges before opening the output.

    Short ranges renew CDN connections. Only network/HTTP transient errors are
    retried; invalid ranges, file errors and content verification errors escape.
    The caller must verify the full SHA256 before promoting the partial file.
    """
    partial = Path(partial)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError(f"Oversized partial file: {partial}")
    retries = 0
    started = time.monotonic()
    while offset < expected_size:
        end = min(expected_size, offset + range_bytes) - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"})
        print(f"DOWNLOAD_RANGE {partial.name}: {offset}-{end}/{expected_size}", flush=True)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as source:
                expected_range = f"bytes {offset}-{end}/{expected_size}"
                if source.status != 206 or source.headers.get("Content-Range") != expected_range:
                    raise ValueError(f"Server did not honor Range: wanted {expected_range}, got status={source.status}, range={source.headers.get('Content-Range')}; output not changed")
                with partial.open("ab") as target:
                    while offset <= end:
                        chunk = source.read(min(1 << 20, end + 1 - offset))
                        if not chunk:
                            raise http.client.IncompleteRead(b"", end + 1 - offset)
                        target.write(chunk)
                        offset += len(chunk)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                raise
            retries += 1
            if retries > max_retries:
                raise
            # Context managers close/flush the file before the next offset is read.
            offset = partial.stat().st_size if partial.exists() else 0
            delay = min(60, 2 ** retries)
            print(f"DOWNLOAD_RETRY {partial.name}: {type(error).__name__}: {error}; retry={retries}/{max_retries}, resume_byte={offset}, delay={delay}s", flush=True)
            time.sleep(delay)
            continue
        print(f"DOWNLOAD_PROGRESS {partial.name}: {offset}/{expected_size} bytes, elapsed={time.monotonic()-started:.1f}s", flush=True)
