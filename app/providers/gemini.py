"""Google Gemini provider implementations for description and caption generation."""

import asyncio
import os
import random
import logging
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

logger = logging.getLogger(__name__)

class GeminiDescriptionProvider(DescriptionProvider):
    """
    Multimodal Gemini provider with Multi-Key Rotation and Model Fallback.
    Designed for Hackathons to bypass daily '20 requests/day' limits.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        # Load multiple keys from environment if available, otherwise use the build-time key
        raw_keys = os.getenv("GEMINI_KEYS", api_key)
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self.current_key_index = 0
        
        # Initialize the first client
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
        
        # Priority list: 1.5 models usually have 1,500/day limits, 
        # while 2.0/2.5/3.5 often have only 20/day on free tier.
        self.model_fallback_list = [
            "gemini-1.5-flash",       # High daily quota (1500/day)
            "gemini-1.5-flash-002",   # Stable 1.5
            "gemini-2.0-flash-001",   # Stable 2.0
            "gemini-2.5-flash",       # High performance 2.5
            "gemini-1.5-pro",         # Last resort (Powerful but slow)
        ]
        
        if model_name in self.model_fallback_list:
            self.model_fallback_list.remove(model_name)
        self.model_fallback_list.insert(0, model_name)

    def _rotate_key(self):
        """Switches to the next available API key in the pool."""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            new_key = self.api_keys[self.current_key_index]
            self.client = genai.Client(api_key=new_key)
            logger.info(f"🔄 API Key exhausted. Rotated to key #{self.current_key_index} (...{new_key[-4:]})")
            return True
        return False

    async def upload_video(self, filepath: str) -> str:
        async def _upload() -> str:
            try:
                uploaded_file = await asyncio.wait_for(
                    asyncio.to_thread(self.client.files.upload, file=filepath),
                    timeout=REQUEST_TIMEOUT,
                )
                return uploaded_file.uri
            except Exception as exc:
                # If upload fails due to key quota, try rotating once
                if "429" in str(exc) and self._rotate_key():
                    return await _upload()
                raise UploadError(f"Video upload failed: {exc}") from exc

        try:
            return await with_retry(_upload)
        except Exception as exc:
            raise UploadError(f"Upload failed after retries: {exc}") from exc

    async def generate_description(
        self,
        video_ref: str | None,
        keyframe_paths: list[str] | None,
    ) -> str:
        # Prepare the multimodal payload
        if video_ref is not None:
            content_parts = [
                {"file_data": {"file_uri": video_ref, "mime_type": "video/mp4"}},
                DESCRIPTION_PROMPT,
            ]
        elif keyframe_paths is not None:
            images = [Image.open(path) for path in keyframe_paths]
            content_parts = [
                "These are sequential keyframes from a video, ordered chronologically.",
                *images,
                DESCRIPTION_PROMPT,
            ]
        else:
            raise GenerationError("Either video_ref or keyframe_paths required.")

        async def _generate_with_resilience() -> str:
            # We will iterate through all keys and all models
            keys_tried = 0
            while keys_tried < len(self.api_keys):
                for model_id in self.model_fallback_list:
                    try:
                        response = await asyncio.wait_for(
                            asyncio.to_thread(
                                self.client.models.generate_content,
                                model=model_id,
                                contents=content_parts
                            ),
                            timeout=REQUEST_TIMEOUT,
                        )
                        return response.text
                    except Exception as exc:
                        err_msg = str(exc).upper()
                        # If hit quota (429) or high demand (503), try next model
                        if any(x in err_msg for x in ["429", "503", "RESOURCE_EXHAUSTED", "LIMIT"]):
                            logger.warning(f"⚠️ Model {model_id} limited on current key. Trying next model...")
                            continue
                        # If model is simply not found, skip it
                        if "404" in err_msg or "NOT_FOUND" in err_msg:
                            continue
                        raise GenerationError(f"Terminal error with {model_id}: {exc}")
                
                # If all models failed for this key, rotate key and try again
                if not self._rotate_key():
                    break
                keys_tried += 1
                logger.info(f"Retrying pipeline with new API key...")

            raise GenerationError("All API keys and all models in the pool have been exhausted.")

        try:
            return await with_retry(_generate_with_resilience)
        except Exception as exc:
            raise GenerationError(f"Stage 1 failed: {exc}") from exc


class GeminiCaptionProvider(CaptionProvider):
    """Text-to-text Gemini provider for Stage 2: description → caption."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash-8b") -> None:
        # Rotation logic for captions
        raw_keys = os.getenv("GEMINI_KEYS", api_key)
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self.current_key_index = 0
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
        self.model_name = model_name

    def _rotate_key(self):
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
            return True
        return False

    async def generate_caption(self, description: str, style: str) -> str:
        prompt = get_caption_prompt(description, style)

        async def _generate() -> str:
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=prompt
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
                return response.text
            except Exception as exc:
                if "429" in str(exc) and self._rotate_key():
                    return await _generate()
                raise GenerationError(f"Caption generation failed: {exc}") from exc

        try:
            return await with_retry(_generate)
        except Exception as exc:
            raise GenerationError(f"Caption failed after retries: {exc}") from exc