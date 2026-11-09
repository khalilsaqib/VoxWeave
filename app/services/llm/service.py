import json
import httpx
import asyncio

from typing import Any, AsyncIterator
from app.config.settings import Settings
from app.config.prompts import prompt_options
from app.core.logger import logger


class LlmService:
    def __init__(
        self,
        settings: Settings,
        system_prompt: str = (
            "You are a helpful conversational assistant. "
            "Respond naturally and conversationally. "
            "Do not use emojis. "
            "Do not use decorative symbols or special characters. "
            "Avoid markdown formatting. "
            "Do not use bullet points, numbered lists, tables, or code blocks unless explicitly requested. "
            "Keep responses concise and suitable for speech synthesis. "
            "Use plain natural language that can be spoken aloud by a voice assistant."
            "If user can ask in any language, you should also reply in user's language"
        ),
    ):
        self.settings = settings
        self.system_prompt = system_prompt

        self._histories: dict[str, list[dict[str, str]]] = {}
        self._history_lock = asyncio.Lock()

        self.voice_rules_prompt = prompt_options.voice_rules_prompt
        logger.info(
            "[LLM Service] | Service created | URL=%s", self.settings.base_llm_url
        )

    async def _get_history(
        self, session_id: str
    ) -> list[dict[str, str]]:

        async with self._history_lock:
            history = self._histories.get(session_id)
            if history is None:
                history = []
                self._histories[session_id] = history
            return list(history)

    async def _add_message(
        self, session_id: str, role: str, content: str
    ):
        async with self._history_lock:
            history = self._histories.setdefault(
                session_id,
                [
                    {
                        "role": "system",
                        "content": (self.system_prompt or ""),
                    }
                ],
            )
            history.append(
                {
                    "role": role,
                    "content": content,
                }
            )
            self._trim_history(history)

    def _trim_history(self, history: list[dict[str, str]]):
        """
        Keep the system prompt and the most recent
        conversation messages.
        """

        system_message = history[0]

        conversation = history[1:]

        if len(conversation) > self.settings.llm_max_history_messages:
            conversation = conversation[-self.settings.llm_max_history_messages :]

        history.clear()
        history.append(system_message)
        history.extend(conversation)

    async def stream(
        self,
        user_text: str,
        is_voice: bool = False,
        temperature: float = 0.9,
    ) -> AsyncIterator[str]:

        user_text = user_text.strip()

        if not user_text:
            return

        if is_voice:
            system_prompt = (
                (self.system_prompt or "")
                + "\n\n"
                + (self.voice_rules_prompt or "")
            )
        else:
            system_prompt = self.system_prompt or ""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

        payload = {
            "model": self.settings.chat_llm_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "temperature": 0.0,
            "max_tokens": 2048
        }

        timeout = httpx.Timeout(
            connect=10.0,
            read=self.settings.http_timeout_seconds,
            write=10.0,
            pool=10.0,
        )

        logger.info(
            "[LLM Service] | Stream | Request started | Messages=%s", len(messages)
        )

        async with httpx.AsyncClient(timeout=timeout) as client:

            try:
                async with client.stream(
                    "POST",
                    f"{self.settings.base_llm_url.rstrip('/')}/{self.settings.chat_endpoint}",
                    json=payload
                ) as response:

                    if response.status_code != 200:

                        error_body = await response.aread()

                        raise RuntimeError(
                            f"LLM request failed | "
                            f"status={response.status_code} | "
                            f"body="
                            f"{error_body.decode(errors='replace')}"
                        )

                    logger.info("[LLM Service] | Stream | Stream started")

                    async for line in response.aiter_lines():

                        if not line or not line.startswith("data:"):
                            continue

                        data = line[len("data:") :].strip()

                        if not data:
                            continue

                        if data == "[DONE]":
                            break

                        token = self.extract_token(data)
                        if not token:
                            continue

                        yield token

                logger.info("[LLM Service] | Stream | Request finished")

            except httpx.TimeoutException as exc:
                logger.error("[LLM Service] | Stream | Timed out")
                raise RuntimeError("LLM request timed out") from exc

            except httpx.HTTPError as exc:
                logger.error("[LLM Service] | Stream | HTTP error")
                raise RuntimeError(f"LLM HTTP error: {exc}") from exc
        

    async def clear_session(
        self,
        session_id: str,
    ):
        async with self._history_lock:
            self._histories.pop(session_id, None)        

        logger.info("[LLM Service] | ClearSession | Session cleared | Session=%s", session_id)

    async def get_history(
        self, session_id: str
    ) -> list[dict[str, str]]:
        return await self._get_history(session_id)
    
    # ---------------------------------------------------------
    # Extract token from one SSE JSON message
    # ---------------------------------------------------------
    @staticmethod
    def extract_token(data: str) -> str | None:
        try:
            chunk: dict[str, Any] = json.loads(data)

            choices = chunk.get("choices") or []

            if not choices:
                return None

            delta = choices[0].get("delta") or {}
            token = delta.get("content")

            return token if isinstance(token, str) else None

        except (json.JSONDecodeError, TypeError, AttributeError):
            return None