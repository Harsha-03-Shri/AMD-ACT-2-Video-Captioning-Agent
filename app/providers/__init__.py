"""Provider factory — builds model pools from config."""

import json
import os

from app.providers.base import (
    CaptionProvider,
    DescriptionProvider,
    GenerationError,
    UploadError,
)
from app.providers.gemini import GeminiCaptionProvider, GeminiDescriptionProvider
from app.providers.groq import GroqCaptionProvider
from app.providers.mistral import MistralCaptionProvider

__all__ = [
    "CaptionProvider",
    "DescriptionProvider",
    "GenerationError",
    "UploadError",
    "load_model_config",
    "build_description_providers",
    "build_caption_providers",
]

# Registry mapping provider names to their classes
_DESCRIPTION_PROVIDERS: dict[str, type[DescriptionProvider]] = {
    "gemini": GeminiDescriptionProvider,
}

_CAPTION_PROVIDERS: dict[str, type[CaptionProvider]] = {
    "gemini": GeminiCaptionProvider,
    "groq": GroqCaptionProvider,
    "mistral": MistralCaptionProvider,
}


def load_model_config(config_path: str = "/workspace/models_config.json") -> dict:
    """Load model configuration from JSON file.

    Args:
        config_path: Path to the models_config.json file.

    Returns:
        Parsed config dict with 'description_models' and 'caption_models'.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config structure is invalid.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    if "description_models" not in config:
        raise ValueError("Config missing 'description_models' key")
    if "caption_models" not in config:
        raise ValueError("Config missing 'caption_models' key")

    return config


def build_description_providers(config: dict) -> list[DescriptionProvider]:
    """Instantiate description providers from config.

    Args:
        config: Full model config dict.

    Returns:
        List of initialized DescriptionProvider instances.
    """
    providers = []
    for entry in config["description_models"]:
        provider_name = entry["provider"].lower()
        model_name = entry.get("model", "gemini-2.0-flash")
        api_key = os.environ.get(entry["api_key_env"], "")

        if not api_key:
            raise ValueError(
                f"Environment variable '{entry['api_key_env']}' is not set "
                f"for description model '{model_name}'"
            )

        cls = _DESCRIPTION_PROVIDERS.get(provider_name)
        if cls is None:
            raise ValueError(
                f"Unknown description provider: '{provider_name}'. "
                f"Available: {list(_DESCRIPTION_PROVIDERS.keys())}"
            )

        providers.append(cls(api_key=api_key, model_name=model_name))

    return providers


def build_caption_providers(
    config: dict,
) -> list[tuple[CaptionProvider, float]]:
    """Instantiate caption providers from config.

    Args:
        config: Full model config dict.

    Returns:
        List of (CaptionProvider, cooldown_seconds) tuples.
    """
    providers = []
    for entry in config["caption_models"]:
        provider_name = entry["provider"].lower()
        model_name = entry.get("model", "")
        api_key = os.environ.get(entry["api_key_env"], "")
        cooldown = entry.get("cooldown_seconds", 5.0)

        if not api_key:
            raise ValueError(
                f"Environment variable '{entry['api_key_env']}' is not set "
                f"for caption model '{model_name}'"
            )

        cls = _CAPTION_PROVIDERS.get(provider_name)
        if cls is None:
            raise ValueError(
                f"Unknown caption provider: '{provider_name}'. "
                f"Available: {list(_CAPTION_PROVIDERS.keys())}"
            )

        providers.append((cls(api_key=api_key, model_name=model_name), cooldown))

    return providers
