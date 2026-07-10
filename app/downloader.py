"""Async video download with retry logic."""

import httpx

from app.config import REQUEST_TIMEOUT
from app.retry import with_retry


async def download_video(task_id: str, video_url: str) -> str:
    """Download a video from a URL to a temporary file.

    Uses httpx with streaming to handle large files efficiently.
    The download is wrapped with retry logic (3 retries, exponential backoff).

    Args:
        task_id: Unique identifier for the task, used in the filename.
        video_url: URL of the video to download.

    Returns:
        The file path where the video was saved (/tmp/<task_id>.mp4).

    Raises:
        httpx.HTTPStatusError: If the server returns an error after all retries.
        httpx.TimeoutException: If the request times out after all retries.
        httpx.RequestError: If a connection error occurs after all retries.
    """
    output_path = f"/tmp/{task_id}.mp4"

    async def _do_download() -> str:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream("GET", video_url) as response:
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
        return output_path

    return await with_retry(_do_download)
