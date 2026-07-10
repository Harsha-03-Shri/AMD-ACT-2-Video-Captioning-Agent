"""Mistral provider for text-to-text caption generation."""

import asyncio

from mistralai import Mistral

from app.config import REQUEST_TIMEOUT
from app.prompts import get_caption_prompt
from app.providers.base import CaptionProvider, GenerationError
from app.retry import with_retry


class MistralCaptionProvider(CaptionProvider):
    """Mistral-hosted model for Stage 2: description → caption."""

    def __init__(self, api_key: str, model_name: str = "mistral-small-latest") -> None:
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name

    async def generate_caption(self, description: str, style: str) -> str:
        prompt = get_caption_prompt(description, style)

        async def _generate() -> str:
            try:
                response = await asyncio.wait_for(
                    self.client.chat.complete_async(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                        temperature=0.7,
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
                return response.choices[0].message.content.strip()
            except asyncio.TimeoutError:
                raise GenerationError(f"Mistral caption timed out after {REQUEST_TIMEOUT}s")
            except GenerationError:
                raise
            except Exception as exc:
                raise GenerationError(f"Mistral caption failed: {exc}") from exc

        try:
            return await with_retry(_generate)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Mistral caption failed after retries: {exc}") from exc
