"""OpenRouter vision-capable model wrapper."""

from __future__ import annotations

import base64
import imghdr
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base import BaseModel


class OpenRouterVisionModel(BaseModel):
    """Model wrapper for hitting OpenRouter's chat completions endpoint."""

    def __init__(
        self,
        model_slug: str,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 300,
        system_prompt: Optional[str] = None,
        force_binary: bool = True,
        positive_aliases: Optional[List[str]] = None,
        negative_aliases: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenRouter client.

        Args:
            model_slug: Identifier exposed by OpenRouter (e.g., ``openai/gpt-4o-mini``).
            api_key: Optional API key. Defaults to OPENROUTER_API_KEY env var.
            base_url: Chat completions endpoint (override for proxies/self-hosting).
            http_referer: Optional Referer header required by some providers.
            x_title: Optional X-Title header for dashboard labeling.
            temperature: Sampling temperature (default 0 for determinism).
            max_output_tokens: Maximum tokens to request from the model.
            system_prompt: Optional instruction string prepended to every request.
            force_binary: If True, map outputs to yes/no using alias lists.
            positive_aliases: Extra strings that should count as yes.
            negative_aliases: Extra strings that should count as no.
            **kwargs: Stored in BaseModel metadata.
        """
        super().__init__(
            model_name="openrouter",
            model_slug=model_slug,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
        self.model_slug = model_slug
        self.base_url = base_url
        self.http_referer = http_referer
        self.x_title = x_title
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_prompt = system_prompt
        self.force_binary = force_binary
        self.positive_aliases = [
            "yes",
            "yeah",
            "yep",
            "affirmative",
            "true",
            "correct",
        ] + (positive_aliases or [])
        self.negative_aliases = [
            "no",
            "nope",
            "nah",
            "negative",
            "false",
            "incorrect",
        ] + (negative_aliases or [])

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set to use OpenRouterVisionModel.")

    def predict(
        self,
        image_path: Optional[str],
        question: str,
        image_bytes: Optional[bytes] = None,
    ) -> Dict[str, str]:
        """Send an image+question pair via OpenRouter and return both outputs."""
        if not question:
            raise ValueError("Question must be a non-empty string.")

        encoded_image, mime_type = self._encode_image(image_path, image_bytes)
        messages = self._build_messages(question, encoded_image, mime_type)
        payload = {
            "model": self.model_slug,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        response = requests.post(self.base_url, json=payload, headers=headers, timeout=120)
        if not response.ok:
            raise RuntimeError(
                f"OpenRouter request failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        raw_text = self._extract_text(data)
        processed = self._postprocess_response(raw_text)
        return {
            "prediction": processed,
            "raw_output": raw_text,
        }

    def _build_messages(
        self, question: str, encoded_image: str, mime_type: str
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_image}"
                        },
                    },
                ],
            }
        )
        return messages

    def _encode_image(
        self, image_path: Optional[str], image_bytes: Optional[bytes]
    ) -> Tuple[str, str]:
        if image_bytes is None:
            if not image_path:
                raise ValueError("Either image_path or image_bytes must be provided.")
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            data = path.read_bytes()
            mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        else:
            data = image_bytes
            mime_type = self._guess_mime_from_bytes(image_bytes)
        encoded = base64.b64encode(data).decode("utf-8")
        if not mime_type.startswith("image/"):
            mime_type = "image/png"
        return encoded, mime_type

    @staticmethod
    def _guess_mime_from_bytes(image_bytes: bytes) -> str:
        kind = imghdr.what(None, h=image_bytes)
        if not kind:
            return "image/png"
        if kind == "jpg":
            kind = "jpeg"
        return f"image/{kind}"

    @staticmethod
    def _extract_text(response_data: Dict[str, Any]) -> str:
        choices = response_data.get("choices")
        if not choices:
            raise RuntimeError("No choices returned from OpenRouter.")
        message = choices[0].get("message")
        if not message:
            raise RuntimeError("No message found in OpenRouter response.")
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = [
                chunk.get("text", "").strip()
                for chunk in content
                if isinstance(chunk, dict) and chunk.get("type") == "text" and chunk.get("text")
            ]
            if texts:
                return " ".join(texts).strip()
        raise RuntimeError("Unable to extract text from OpenRouter response.")

    def _postprocess_response(self, text: str) -> str:
        normalized = text.strip()
        if not self.force_binary:
            return normalized

        lowered = normalized.lower()
        candidate = self._match_alias(lowered)
        if candidate:
            return candidate

        first_token = lowered.split()[0]
        candidate = self._match_alias(first_token)
        if candidate:
            return candidate

        return normalized

    def _match_alias(self, token: str) -> Optional[str]:
        token = token.strip(".,!?")
        if token in self.positive_aliases:
            return "yes"
        if token in self.negative_aliases:
            return "no"
        return None
