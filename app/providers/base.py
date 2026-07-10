"""Abstract base classes for LLM providers."""

from abc import ABC, abstractmethod


class UploadError(Exception):
    """Raised when video upload to the LLM provider fails."""
    pass


class GenerationError(Exception):
    """Raised when text generation via the LLM provider fails."""
    pass


class DescriptionProvider(ABC):
    """Abstract base for multimodal providers that generate video descriptions.

    These providers accept video/keyframe input and produce text descriptions.
    """

    @abstractmethod
    async def upload_video(self, filepath: str) -> str:
        """Upload a video file for analysis. Returns a provider-specific reference."""
        ...

    @abstractmethod
    async def generate_description(
        self,
        video_ref: str | None,
        keyframe_paths: list[str] | None,
    ) -> str:
        """Generate an objective video description from video or keyframes."""
        ...


class CaptionProvider(ABC):
    """Abstract base for text-to-text providers that generate styled captions.

    These providers accept a text description and produce styled captions.
    """

    @abstractmethod
    async def generate_caption(self, description: str, style: str) -> str:
        """Generate a styled caption from a text description."""
        ...
