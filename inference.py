"""LibertAI inference client for agent VMs."""

import asyncio
import logging

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Retry config for transient API errors
_MAX_RETRIES = 2
_BASE_DELAY = 2.0  # seconds, doubles on each retry

# Errors worth retrying (transient / server-side)
_RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class InferenceClient:
    """Thin wrapper around AsyncOpenAI pointed at LibertAI."""

    def __init__(self, api_key: str, base_url: str = "https://api.libertai.io/v1"):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=180.0,  # per-request timeout for the HTTP call
            max_retries=0,  # we handle retries ourselves for better logging
        )

    async def chat(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
    ):
        """Send a chat completion request with retry for transient errors.

        Returns the full message object from the first choice.
        Raises on non-retryable errors or after all retries are exhausted.
        """
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                return response.choices[0].message
            except _RETRYABLE as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Inference attempt {attempt + 1} failed ({type(e).__name__}), "
                        f"retrying in {delay:.0f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Inference failed after {_MAX_RETRIES + 1} attempts: {e}"
                    )

        raise last_error  # type: ignore[misc]
