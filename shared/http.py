import time
from contextlib import contextmanager
from typing import Iterator

import httpx


DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)


def service_client() -> httpx.Client:
    return httpx.Client(timeout=DEFAULT_TIMEOUT)


@contextmanager
def timer_ms() -> Iterator[callable]:
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000
