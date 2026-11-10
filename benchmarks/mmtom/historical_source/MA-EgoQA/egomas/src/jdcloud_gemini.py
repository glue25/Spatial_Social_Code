"""Minimal Gemini-compatible client for the provisioned JD Cloud endpoint."""

from __future__ import annotations

import base64
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_CREDENTIALS_FILE = Path(
    os.getenv("GEMINI_CREDENTIALS_FILE", "credentials.env")
)


class GeminiServiceError(RuntimeError):
    """Raised when the remote model service cannot return usable text."""


@dataclass(frozen=True)
class GeminiServiceConfig:
    api_key: str
    api_base: str
    model: str


def _export_value(text: str, name: str) -> str:
    match = re.search(
        rf"^\s*export\s+{re.escape(name)}\s*=\s*([\"'])(.*?)\1",
        text,
        flags=re.MULTILINE,
    )
    if match:
        return match.group(2).strip()
    match = re.search(
        rf"^\s*export\s+{re.escape(name)}\s*=\s*([^\s;]+)",
        text,
        flags=re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def load_service_config(
    credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE,
    model_override: str | None = None,
) -> GeminiServiceConfig:
    """Load config from environment, optionally falling back to an env file."""
    values = {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "").strip(),
        "GEMINI_API_BASE": os.getenv("GEMINI_API_BASE", "").strip(),
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "").strip(),
    }
    path = Path(credentials_file)
    if not all(values.values()) and path.is_file():
        text = path.read_text(encoding="utf-8")
        for name in values:
            values[name] = values[name] or _export_value(text, name)
    model = (model_override or values["GEMINI_MODEL"]).strip()
    if not values["GEMINI_API_KEY"] or not values["GEMINI_API_BASE"] or not model:
        raise RuntimeError(
            "Gemini service config is incomplete; provide GEMINI_API_KEY, "
            "GEMINI_API_BASE, and GEMINI_MODEL or a credentials file"
        )
    return GeminiServiceConfig(
        api_key=values["GEMINI_API_KEY"],
        api_base=values["GEMINI_API_BASE"],
        model=model,
    )


def _extract_text(payload: dict[str, Any]) -> str:
    texts = []
    for candidate in payload.get("candidates", []) or []:
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            if part.get("text"):
                texts.append(str(part["text"]))
    return "\n".join(texts).strip()


class JDCloudGeminiClient:
    def __init__(
        self,
        config: GeminiServiceConfig,
        timeout_seconds: int = 180,
        max_attempts: int = 5,
    ) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def generate_parts(
        self, parts: list[dict[str, Any]], max_output_tokens: int = 4096
    ) -> str:
        payload = {
            "model": self.config.model,
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max_output_tokens,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = requests.post(
                    self.config.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 200:
                    text = _extract_text(response.json())
                    if not text:
                        raise GeminiServiceError("response contained no candidate text")
                    return text
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (requests.RequestException, ValueError, GeminiServiceError) as error:
                last_error = str(error)
            if attempt < self.max_attempts:
                time.sleep(min(30, 2 ** (attempt - 1)))
        raise GeminiServiceError(
            f"Gemini request failed after {self.max_attempts} attempts: {last_error}"
        )

    def generate(self, prompt: str, max_output_tokens: int = 4096) -> str:
        return self.generate_parts(
            [{"text": prompt}], max_output_tokens=max_output_tokens
        )

    def generate_video(
        self,
        prompt: str,
        video_path: str | Path,
        max_output_tokens: int = 4096,
    ) -> str:
        """Send a local MP4 as a native Gemini inline-data part."""
        encoded = base64.b64encode(Path(video_path).read_bytes()).decode("ascii")
        return self.generate_parts(
            [
                {"inline_data": {"mimeType": "video/mp4", "data": encoded}},
                {"text": prompt},
            ],
            max_output_tokens=max_output_tokens,
        )
