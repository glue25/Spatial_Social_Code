"""Seekable HTTP Range reader for large remote media files."""

from __future__ import annotations

import io
import time
from collections import OrderedDict

import requests
from requests import RequestException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HTTPRangeReader(io.RawIOBase):
    """Expose a remote byte-range resource as a cached seekable file object."""

    def __init__(
        self,
        url: str,
        size: int,
        block_size: int = 1 << 16,
        max_blocks: int = 256,
        timeout: float = 30.0,
    ) -> None:
        super().__init__()
        if size <= 0 or block_size <= 0 or max_blocks <= 0:
            raise ValueError("size, block_size, and max_blocks must be positive")
        self.url = url
        self.size = size
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.timeout = timeout
        self.position = 0
        self.bytes_downloaded = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._session = requests.Session()
        retry = Retry(
            total=1,
            backoff_factor=1.0,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise OSError("negative seek position")
        self.position = min(position, self.size)
        return self.position

    def _fetch_block(self, index: int) -> bytes:
        start = index * self.block_size
        end = min(self.size, start + self.block_size) - 1
        response = None
        error: Exception | None = None
        for attempt in range(6):
            try:
                response = self._session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(10.0, self.timeout),
                )
                break
            except RequestException as exc:
                error = exc
                if response is not None:
                    response.close()
                    response = None
                if attempt < 5:
                    time.sleep(min(1.0 + attempt, 5.0))
        if response is None:
            raise OSError(f"failed HTTP Range {start}-{end} after 6 attempts") from error
        if response.status_code != 206:
            response.close()
            raise OSError(
                f"server did not honor HTTP Range {start}-{end}: "
                f"status {response.status_code}"
            )
        data = response.content
        expected = end - start + 1
        if len(data) != expected:
            raise OSError(
                f"short HTTP Range response {start}-{end}: {len(data)} != {expected}"
            )
        self.bytes_downloaded += len(data)
        return data

    def _block(self, index: int) -> bytes:
        if index in self._cache:
            data = self._cache.pop(index)
            self._cache[index] = data
            return data
        data = self._fetch_block(index)
        self._cache[index] = data
        while len(self._cache) > self.max_blocks:
            self._cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        output = bytearray()
        while len(output) < size:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            take = min(size - len(output), len(block) - offset)
            if take <= 0:
                break
            output.extend(block[offset : offset + take])
            self.position += take
        return bytes(output)

    def close(self) -> None:
        if not self.closed:
            self._session.close()
            self._cache.clear()
        super().close()
