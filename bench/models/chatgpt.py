"""ChatGPT Vision API model wrapper."""

from __future__ import annotations

import base64
import imghdr
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .base import BaseModel


class ChatGPTVisionModel(BaseModel):
    """Model wrapper that proxies image+text prompts to the ChatGPT VLM API."""

    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 300,
        system_prompt: Optional[str] = None,
        force_binary: bool = True,
        positive_aliases: Optional[List[str]] = None,
        negative_aliases: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ChatGPT vision model client.

        Args:
            openai_model: Name of the OpenAI vision-capable model to call.
            api_key: Optional API key. Defaults to OPENAI_API_KEY env var.
            api_base: Optional custom API base URL (for Azure/OpenAI proxies).
            temperature: Sampling temperature. Defaults to 0 for determinism.
            max_output_tokens: Upper bound on generated tokens per call.
            system_prompt: Optional instruction string prepended to each request.
            **kwargs: Additional configuration persisted in BaseModel metadata.
        """
        super().__init__(
            model_name="chatgpt-vision",
            openai_model=openai_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
        self.openai_model = openai_model
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

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY must be set to use ChatGPTVisionModel."
            )

        client_kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        self.client = OpenAI(**client_kwargs)

    def predict(
        self,
        image_path: Optional[str],
        question: str,
        image_bytes: Optional[bytes] = None,
    ) -> str:
        """Send an image+question pair to ChatGPT and return the response text."""
        if not question:
            raise ValueError("Question must be a non-empty string.")

        encoded_image, mime_type = self._encode_image(image_path, image_bytes)
        messages = self._build_messages(question, encoded_image, mime_type)
        response = self.client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        raw_text = self._extract_text(response)
        processed = self._postprocess_response(raw_text)
        return {
            "prediction": processed,
            "raw_output": raw_text,
        }

    def _build_messages(
        self, question: str, encoded_image: str, mime_type: str
    ) -> List[Dict[str, Any]]:
        """Construct chat completion messages for the API request."""
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
        self,
        image_path: Optional[str],
        image_bytes: Optional[bytes],
    ) -> Tuple[str, str]:
        """Encode either a local file or in-memory bytes into base64."""
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
        """Best-effort MIME detection for raw bytes."""
        kind = imghdr.what(None, h=image_bytes)
        if not kind:
            return "image/png"
        if kind == "jpg":
            kind = "jpeg"
        return f"image/{kind}"

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract assistant text from a chat.completions response."""
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("No choices returned from ChatGPT.")
        message = choices[0].message
        if not message:
            raise RuntimeError("No message found in ChatGPT response.")
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
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
        raise RuntimeError("Unable to extract text from ChatGPT response.")

    def _postprocess_response(self, text: str) -> str:
        """Map ChatGPT's free-form response to yes/no when requested."""
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

        # fall back to raw response when no alias matched
        return normalized

    def _match_alias(self, token: str) -> Optional[str]:
        token = token.strip(".,!?")
        if token in self.positive_aliases:
            return "yes"
        if token in self.negative_aliases:
            return "no"
        return None
