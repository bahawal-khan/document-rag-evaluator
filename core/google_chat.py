"""LangChain-compatible Gemini chat adapter that never mutates process env vars."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field, SecretStr


class SessionGoogleChatModel(BaseChatModel):
    """Minimal Google GenAI chat model using an explicitly supplied session key."""

    model_name: str
    api_key: SecretStr
    temperature: float = 0.0
    max_output_tokens: int | None = None
    timeout: float = 90.0
    streaming: bool = True
    client: Any = Field(default=None, exclude=True, repr=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        from google import genai
        from google.genai import types

        self.client = genai.Client(
            api_key=self.api_key.get_secret_value(),
            http_options=types.HttpOptions(timeout=max(1, int(self.timeout * 1000))),
        )

    @property
    def _llm_type(self) -> str:
        return "session-google-genai"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _contents_and_config(self, messages: list[BaseMessage]):
        from google.genai import types

        contents = []
        system_parts = []
        for message in messages:
            text = message.content if isinstance(message.content, str) else str(message.content)
            if isinstance(message, SystemMessage):
                system_parts.append(text)
                continue
            role = "model" if isinstance(message, AIMessage) else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            system_instruction="\n\n".join(system_parts) or None,
        )
        return contents, config

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        contents, config = self._contents_and_config(messages)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response.text or ""))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        contents, config = self._contents_and_config(messages)
        response = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        for part in response:
            text = part.text or ""
            if not text:
                continue
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        contents, config = self._contents_and_config(messages)
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response.text or ""))])

    def close(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()

