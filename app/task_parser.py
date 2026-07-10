"""JSON input parsing and validation for task files."""

import json

from app.models import Task
from app.config import INPUT_PATH


def parse_tasks(filepath: str) -> list[Task]:
    """Parse and validate a JSON task file.

    Reads the file at filepath, parses it as JSON, validates it is an array
    of task objects each containing task_id (string), video_url (string),
    and styles (non-empty list of strings).

    Args:
        filepath: Path to the JSON task file.

    Returns:
        A list of validated Task objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the JSON structure is invalid or tasks are malformed.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Task file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in task file: {filepath}", e.doc, e.pos
        )

    if not isinstance(data, list):
        raise ValueError(
            f"Task file must contain a JSON array, got {type(data).__name__}"
        )

    tasks: list[Task] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Task at index {index} must be a JSON object, got {type(item).__name__}"
            )

        # Validate task_id
        if "task_id" not in item:
            raise ValueError(f"Task at index {index} is missing required field 'task_id'")
        if not isinstance(item["task_id"], str):
            raise ValueError(
                f"Task at index {index}: 'task_id' must be a string, "
                f"got {type(item['task_id']).__name__}"
            )

        # Validate video_url
        if "video_url" not in item:
            raise ValueError(f"Task at index {index} is missing required field 'video_url'")
        if not isinstance(item["video_url"], str):
            raise ValueError(
                f"Task at index {index}: 'video_url' must be a string, "
                f"got {type(item['video_url']).__name__}"
            )

        # Validate styles
        if "styles" not in item:
            raise ValueError(f"Task at index {index} is missing required field 'styles'")
        if not isinstance(item["styles"], list):
            raise ValueError(
                f"Task at index {index}: 'styles' must be a list, "
                f"got {type(item['styles']).__name__}"
            )
        if len(item["styles"]) == 0:
            raise ValueError(f"Task at index {index}: 'styles' must be a non-empty list")
        for style_index, style in enumerate(item["styles"]):
            if not isinstance(style, str):
                raise ValueError(
                    f"Task at index {index}: 'styles[{style_index}]' must be a string, "
                    f"got {type(style).__name__}"
                )

        tasks.append(
            Task(
                task_id=item["task_id"],
                video_url=item["video_url"],
                styles=item["styles"],
            )
        )

    return tasks
