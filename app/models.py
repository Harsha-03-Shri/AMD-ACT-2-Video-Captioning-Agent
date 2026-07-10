"""Data models for the Video Captioning Agent."""

from dataclasses import dataclass


@dataclass
class Task:
    """A single video captioning task parsed from the input file."""

    task_id: str
    video_url: str
    styles: list[str]


@dataclass
class TaskResult:
    """The result of processing a single task, mapping styles to captions."""

    task_id: str
    captions: dict[str, str]
