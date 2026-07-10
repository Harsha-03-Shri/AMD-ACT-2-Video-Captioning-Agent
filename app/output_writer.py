"""Result serialization to JSON output file."""

import json

from app.models import TaskResult


def write_results(results: list[TaskResult], filepath: str) -> None:
    """Serialize results to JSON and write to filepath.

    Produces a JSON array where each element has the structure:
    {"task_id": "...", "captions": {"style": "caption text", ...}}

    Args:
        results: List of TaskResult objects to serialize.
        filepath: Path to write the JSON output file.
    """
    output = [
        {"task_id": result.task_id, "captions": result.captions}
        for result in results
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2,ensure_ascii=False)
