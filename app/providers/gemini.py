"""Google Gemini provider implementations for description and caption generation."""

import asyncio
import os
from google import genai
from PIL import Image

from app.config import REQUEST_TIMEOUT
from app.prompts import DESCRIPTION_PROMPT, get_caption_prompt
from app.providers.base import (
    CaptionProvider,
    DescriptionProvider,
    GenerationError,
    UploadError,
)
from app.retry import with_retry


class GeminiDescriptionProvider(DescriptionProvider):
    """Multimodal Gemini provider for Stage 1: keyframes/video → description."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        # NEW: The Client object is required for keys starting with 'AQ.'
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def upload_video(self, filepath: str) -> str:
        async def _upload() -> str:
            try:
                # NEW: Use client.files.upload instead of genai.upload_file
                uploaded_file = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.files.upload, 
                        file=filepath
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
                # The returned object has a 'uri' property
                return uploaded_file.uri
            except asyncio.TimeoutError:
                raise UploadError(f"Video upload timed out after {REQUEST_TIMEOUT}s")
            except Exception as exc:
                raise UploadError(f"Video upload failed: {exc}") from exc

        try:
            return await with_retry(_upload)
        except UploadError:
            raise
        except Exception as exc:
            raise UploadError(f"Upload failed after retries: {exc}") from exc

    async def generate_description(
        self,
        video_ref: str | None,
        keyframe_paths: list[str] | None,
    ) -> str:
        async def _generate() -> str:
            try:
                if video_ref is not None:
                    # NEW: Structure for File URIs in the new SDK
                    content_parts = [
                        {"file_data": {"file_uri": video_ref, "mime_type": "video/mp4"}},
                        DESCRIPTION_PROMPT,
                    ]
                elif keyframe_paths is not None:
                    # Open images as PIL objects
                    images = [Image.open(path) for path in keyframe_paths]
                    content_parts = [
                        "These are sequential keyframes from a video, ordered chronologically.",
                        *images,
                        DESCRIPTION_PROMPT,
                    ]
                else:
                    raise GenerationError("Either video_ref or keyframe_paths required.")

                # NEW: Use client.models.generate_content
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=content_parts
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
                return response.text
            except asyncio.TimeoutError:
                raise GenerationError(f"Description timed out after {REQUEST_TIMEOUT}s")
            except GenerationError:
                raise
            except Exception as exc:
                # Catching specific 429/400 errors from the new SDK
                raise GenerationError(f"Description generation failed: {exc}") from exc

        try:
            return await with_retry(_generate)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Description failed after retries: {exc}") from exc


class GeminiCaptionProvider(CaptionProvider):
    """Text-to-text Gemini provider for Stage 2: description → caption."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-lite-001") -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def generate_caption(self, description: str, style: str) -> str:
        prompt = get_caption_prompt(description, style)

        async def _generate() -> str:
            try:
                # NEW: Use client.models.generate_content
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=prompt
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
                return response.text
            except asyncio.TimeoutError:
                raise GenerationError(f"Caption timed out after {REQUEST_TIMEOUT}s")
            except GenerationError:
                raise
            except Exception as exc:
                raise GenerationError(f"Caption generation failed: {exc}") from exc

        try:
            return await with_retry(_generate)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Caption failed after retries: {exc}") from exc