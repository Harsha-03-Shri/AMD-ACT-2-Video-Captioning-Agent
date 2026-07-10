"""Configuration constants for the Video Captioning Agent."""

import os

# Retry settings
MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 2.0

# Timeouts (seconds)
REQUEST_TIMEOUT: int = 30
RUNTIME_TIMEOUT: int = 480  # 8 minutes

# Keyframe extraction
KEYFRAME_RATE: str = "1/4"  # 1 frame every 4 seconds

# File paths
INPUT_PATH: str = "/input/tasks.json"
OUTPUT_PATH: str = "/output/results.json"
MODEL_CONFIG_PATH: str = "/workspace/models_config.json"

# Fallback
FALLBACK_CAPTION: str = "Unable to generate caption"
