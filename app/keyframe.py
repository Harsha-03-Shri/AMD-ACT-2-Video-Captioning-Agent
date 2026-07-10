"""Keyframe extraction fallback using ffmpeg."""

import asyncio
import glob

from app.config import KEYFRAME_RATE


async def extract_keyframes(video_path: str, task_id: str) -> list[str]:
    """Extract 10-15 keyframes from a video using ffmpeg.

    Uses ffmpeg subprocess to extract frames at the configured rate
    (KEYFRAME_RATE = "1/4", i.e., 1 frame every 4 seconds), limited to
    a maximum of 15 frames.

    Args:
        video_path: Path to the input video file.
        task_id: Unique task identifier used for naming output frames.

    Returns:
        Sorted list of extracted keyframe image file paths.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero exit code.
    """
    output_pattern = f"/tmp/{task_id}_frame_%03d.jpg"

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={KEYFRAME_RATE}",
        "-frames:v", "15",
        output_pattern,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed with exit code {process.returncode}: {error_msg}"
        )

    frame_paths = sorted(glob.glob(f"/tmp/{task_id}_frame_*.jpg"))
    return frame_paths
