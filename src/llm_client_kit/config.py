"""Configuration for an OpenAI-shape endpoint.

Only two environment variables exist, deliberately: OPENAI_BASE_URL and
OPENAI_API_KEY. Never per-provider names. Swapping between OpenAI, a local
vLLM server, Ollama, OpenRouter or Groq is then a base_url change with zero
code change -- which is the whole point of coding against the shape rather
than the vendor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _load_env_file(path: Path) -> dict[str, str]:
    """Minimal .env reader. No dependency, no interpolation, no surprises."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


@dataclass(frozen=True)
class ClientConfig:
    """Resolved endpoint settings.

    Resolution order, first hit wins:
        1. explicit arguments
        2. process environment
        3. ./.env
        4. $MAWORKS_SECRETS/.env.shared

    One shared secret store rather than a copy per project: N copies of the
    same key means N chances to commit it and one bad day when it rotates.
    """

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout_s: float = 60.0
    max_retries: int = 3
    max_concurrency: int = 8
    # Deadline for a whole logical operation, propagated across retries so
    # a caller's budget is never silently exceeded by the retry loop.
    deadline_s: float | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: object) -> ClientConfig:
        env: dict[str, str] = {}
        shared = os.environ.get("MAWORKS_SECRETS")
        if shared:
            env.update(_load_env_file(Path(shared) / ".env.shared"))
        env.update(_load_env_file(Path(".env")))
        env.update({k: v for k, v in os.environ.items() if k.startswith("OPENAI_")})

        base = {
            "base_url": env.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            "api_key": env.get("OPENAI_API_KEY", ""),
        }
        if "OPENAI_MODEL" in env:
            base["model"] = env["OPENAI_MODEL"]
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)  # type: ignore[arg-type]

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.extra_headers)
        return h

    def redacted(self) -> dict[str, object]:
        """Safe for logs. The api_key is never returned in full."""
        return {
            "base_url": self.base_url,
            "api_key": f"...{self.api_key[-4:]}" if self.api_key else "(unset)",
            "model": self.model,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "max_concurrency": self.max_concurrency,
        }
