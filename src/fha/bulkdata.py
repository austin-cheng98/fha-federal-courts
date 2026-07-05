"""
Streaming reader for CourtListener bulk-data .csv.bz2 dumps. Resumes on
connection reset via HTTP Range, and parses the Postgres COPY CSV dialect
(escapechar='\\', doublequote=False).
"""
from __future__ import annotations

import bz2
import csv
import io
import time
from typing import Iterator

import requests

BASE = "https://com-courtlistener-storage.s3.us-west-2.amazonaws.com/bulk-data"
DEFAULT_DUMP = "2026-03-31"   # latest quarterly dump as of 2026-06
csv.field_size_limit(10 ** 9)  # opinion text fields are huge

_RETRYABLE = (requests.exceptions.ChunkedEncodingError,
              requests.exceptions.ConnectionError,
              requests.exceptions.ReadTimeout)


class BulkStream(io.RawIOBase):
    """Decompressed byte stream over a bulk .bz2 file that resumes on resets."""

    def __init__(self, url: str, chunk: int = 1 << 20, max_retries: int = 12,
                 on_retry=None):
        self.url = url
        self.chunk = chunk
        self.max_retries = max_retries
        self.on_retry = on_retry
        self.session = requests.Session()
        self.pos = 0                       # compressed bytes pulled so far
        self.dec = bz2.BZ2Decompressor()
        self.out = b""                     # pending decompressed bytes
        self._open(0)

    def _open(self, start: int) -> None:
        headers = {"Range": f"bytes={start}-"} if start else {}
        self.resp = self.session.get(self.url, headers=headers, stream=True,
                                     timeout=300)
        self.resp.raise_for_status()
        self.it = self.resp.iter_content(self.chunk)

    def _raw_next(self) -> bytes | None:
        tries = 0
        while True:
            try:
                return next(self.it)
            except StopIteration:
                return None
            except _RETRYABLE as e:
                tries += 1
                if tries > self.max_retries:
                    raise
                wait = min(2 ** tries, 30)
                if self.on_retry:
                    self.on_retry(self.pos, tries, e)
                time.sleep(wait)
                self._open(self.pos)       # resume from last byte received

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self.out:
            chunk = self._raw_next()
            if chunk is None:
                return 0
            self.pos += len(chunk)
            self.out = self.dec.decompress(chunk)
        n = min(len(b), len(self.out))
        b[:n], self.out = self.out[:n], self.out[n:]
        return n


def stream_bulk_rows(table: str, dump: str = DEFAULT_DUMP,
                     on_retry=None) -> Iterator[list[str]]:
    """Yield CSV rows (lists) from a bulk table, resilient + correct dialect.
    Skips the header row."""
    url = f"{BASE}/{table}-{dump}.csv.bz2"
    stream = BulkStream(url, on_retry=on_retry)
    text = io.TextIOWrapper(io.BufferedReader(stream, buffer_size=1 << 20),
                            encoding="utf-8", errors="replace", newline="")
    reader = csv.reader(text, doublequote=False, escapechar="\\")
    next(reader, None)
    yield from reader


def read_local_bulk_rows(path: str) -> Iterator[list[str]]:
    """Yield CSV rows from a LOCAL bulk .csv.bz2 file (no network, fully reliable).
    Same correct dialect. Use after a resumable disk download of the dump."""
    with bz2.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, doublequote=False, escapechar="\\")
        next(reader, None)
        yield from reader
