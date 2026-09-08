"""Latency instrumentation utilities."""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class Timer:
    """Context manager for measuring wall-clock latency."""
    name: str = "operation"
    _elapsed_ms: float = field(default=0.0, init=False, repr=False)
    _start: float = field(default=0.0, init=False, repr=False)

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self._elapsed_ms = (time.perf_counter() - self._start) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self._elapsed_ms

    def __repr__(self) -> str:
        return f"Timer(name={self.name!r}, elapsed_ms={self._elapsed_ms:.3f})"


@contextmanager
def latency_ms(name: str = "operation") -> Generator[Timer, None, None]:
    """Context manager yielding a Timer that records elapsed milliseconds."""
    t = Timer(name=name)
    with t:
        yield t
