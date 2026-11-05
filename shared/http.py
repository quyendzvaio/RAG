import time
from contextlib import contextmanager
from typing import Callable, Iterator

import httpx

from shared.tracing import trace_headers


DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)
# Mot lan /chat cua llm-service co the gom hai lan goi provider (goi lai khi 502/503/504), moi lan toi DEFAULT_TIMEOUT.
LLM_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
# Mot cau tra loi cua vector/wiki gom toi hai lan /chat (viet lai hoac chon trang, roi tra loi).
ANSWER_TIMEOUT = httpx.Timeout(connect=5.0, read=660.0, write=30.0, pool=5.0)


def service_client(timeout: httpx.Timeout = DEFAULT_TIMEOUT) -> httpx.Client:
    """Client goi service khac; gan header trace LangSmith de service nhan noi tiep cung mot trace."""
    return httpx.Client(timeout=timeout, event_hooks={"request": [_add_trace_headers]})


def _add_trace_headers(request: httpx.Request) -> None:
    request.headers.update(trace_headers())


@contextmanager
def timer_ms() -> Iterator[Callable[[], float]]:
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000
