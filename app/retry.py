"""Shared retry logic with exponential backoff."""

import asyncio
from typing import Awaitable, Callable, TypeVar

from app.config import MAX_RETRIES, RETRY_BASE_DELAY

T = TypeVar("T")


async def with_retry(
    fn: Callable[..., Awaitable[T]],
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
) -> T:
    """Execute an async callable with exponential backoff retry.

    Calls fn() up to max_retries + 1 times total (1 initial attempt + max_retries retries).
    Delays between attempts follow exponential backoff: base_delay * 2^(attempt-1),
    yielding delays of 2s, 4s, 8s for the default base_delay of 2.0.

    Args:
        fn: An async callable (no arguments) to execute.
        max_retries: Maximum number of retry attempts after the initial call.
        base_delay: Base delay in seconds for exponential backoff.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_exception: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exception = exc
            # If we still have retries left, wait with exponential backoff
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

    # This should never be reached without last_exception being set,
    # but raise explicitly for type safety.
    raise last_exception  # type: ignore[misc]
