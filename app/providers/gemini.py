import asyncio
import os
import logging
import time # Added for state checking if needed, though asyncio is preferred
from google import genai
from PIL import Image

from app.config import REQUEST_TIMEOUT
from app.prompts import DESCRIPTION_PROMPT
from app.providers.base import (
    DescriptionProvider,
    GenerationError,
    UploadError,
)
from app.retry import with_retry

logger = logging.getLogger(__name__)

class GeminiDescriptionProvider(DescriptionProvider):
    """
    Multimodal Gemini provider with Multi-Key Rotation and Model Fallback.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        raw_keys = os.getenv("GEMINI_KEYS", api_key)
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self.current_key_index = 0
        
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
        
        self.model_fallback_list = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-002",
            "gemini-2.0-flash-001",
            "gemini-2.5-flash",
            "gemini-1.5-pro",
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
            logger.info(f"🔄 API Key exhausted. Rotated to key #{self.current_key_index}")
            return True
        return False

    async def upload_video(self, filepath: str) -> str:
        """Uploads video and waits for ACTIVE state before returning URI."""
        async def _upload() -> str:
            try:
                # 1. Perform the initial upload
                uploaded_file = await asyncio.wait_for(
                    asyncio.to_thread(self.client.files.upload, path=filepath),
                    timeout=REQUEST_TIMEOUT,
                )
                file_name = uploaded_file.name
                logger.info(f"📤 Uploaded file {file_name}, waiting for ACTIVE state...")

                # 2. Poll for ACTIVE state
                # We use a loop to check the file status every 2 seconds
                while True:
                    file_info = await asyncio.to_thread(self.client.files.get, name=file_name)
                    state = file_info.state.name # Accesses the Enum name (e.g., 'ACTIVE')

                    if state == "ACTIVE":
                        logger.info(f"✅ File {file_name} is ACTIVE and ready.")
                        return file_info.uri
                    
                    if state == "FAILED":
                        raise UploadError(f"File {file_name} failed processing.")

                    # Wait before checking again to avoid spamming the API
                    await asyncio.sleep(2)

            except Exception as exc:
                # If upload/polling fails due to key quota, try rotating
                if any(x in str(exc) for x in ["429", "RESOURCE_EXHAUSTED"]) and self._rotate_key():
                    return await _upload()
                raise UploadError(f"Video processing failed: {exc}") from exc

        try:
            return await with_retry(_upload)
        except Exception as exc:
            raise UploadError(f"Upload/Processing failed after retries: {exc}") from exc

    async def generate_description(
        self,
        video_ref: str | None,
        keyframe_paths: list[str] | None,
    ) -> str:
        # Prepare the multimodal payload
        if video_ref is not None:
            # video_ref is now guaranteed to be ACTIVE because of the logic in upload_video
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
                        # If hit quota or high demand, try next model
                        if any(x in err_msg for x in ["429", "503", "RESOURCE_EXHAUSTED", "LIMIT"]):
                            logger.warning(f"⚠️ Model {model_id} limited. Trying next fallback...")
                            continue
                        
                        # Handle the specific File Not Active error if polling failed or was bypassed
                        if "FAILED_PRECONDITION" in err_msg and "NOT IN AN ACTIVE STATE" in err_msg:
                            logger.error(f"❌ Critical: File not active for model {model_id}.")
                            # You could theoretically add another wait here, but it's handled in upload_video
                        
                        if "404" in err_msg or "NOT_FOUND" in err_msg:
                            continue
                        raise GenerationError(f"Terminal error with {model_id}: {exc}")
                
                if not self._rotate_key():
                    break
                keys_tried += 1
                logger.info(f"Retrying pipeline with new API key...")

            raise GenerationError("All API keys and models exhausted.")

        try:
            return await with_retry(_generate_with_resilience)
        except Exception as exc:
            raise GenerationError(f"Stage 1 failed: {exc}") from exc