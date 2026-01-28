from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import json
import os
import types
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Protocol, TypeVar

import httpx

from vibe.core.llm.exceptions import BackendErrorBuilder
from vibe.core.types import (
    AvailableTool,
    FunctionCall,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StrToolChoice,
    ToolCall,
)
from vibe.core.utils import async_generator_retry, async_retry

if TYPE_CHECKING:
    from vibe.core.config import ModelConfig, ProviderConfig


class PreparedRequest(NamedTuple):
    endpoint: str
    headers: dict[str, str]
    body: bytes


class APIAdapter(Protocol):
    endpoint: ClassVar[str]

    def prepare_request(
        self,
        *,
        model_name: str,
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        enable_streaming: bool,
        provider: ProviderConfig,
        api_key: str | None = None,
        previous_response_id: str | None = None,
    ) -> PreparedRequest: ...

    def parse_response(
        self, data: dict[str, Any], provider: ProviderConfig
    ) -> LLMChunk: ...


BACKEND_ADAPTERS: dict[str, APIAdapter] = {}

T = TypeVar("T", bound=APIAdapter)


def register_adapter(
    adapters: dict[str, APIAdapter], name: str
) -> Callable[[type[T]], type[T]]:

    def decorator(cls: type[T]) -> type[T]:
        adapters[name] = cls()
        return cls

    return decorator


RESPONSES_ADAPTERS: dict[str, APIAdapter] = {}


@register_adapter(BACKEND_ADAPTERS, "openai")
class OpenAIAdapter(APIAdapter):
    endpoint: ClassVar[str] = "/chat/completions"

    def build_payload(
        self,
        model_name: str,
        converted_messages: list[dict[str, Any]],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
    ) -> dict[str, Any]:
        payload = {
            "model": model_name,
            "messages": converted_messages,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = [tool.model_dump(exclude_none=True) for tool in tools]
        if tool_choice:
            payload["tool_choice"] = (
                tool_choice
                if isinstance(tool_choice, str)
                else tool_choice.model_dump()
            )
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return payload

    def build_headers(self, api_key: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _reasoning_to_api(
        self, msg_dict: dict[str, Any], field_name: str
    ) -> dict[str, Any]:
        if field_name != "reasoning_content" and "reasoning_content" in msg_dict:
            msg_dict[field_name] = msg_dict.pop("reasoning_content")
        return msg_dict

    def _reasoning_from_api(
        self, msg_dict: dict[str, Any], field_name: str
    ) -> dict[str, Any]:
        if field_name != "reasoning_content" and field_name in msg_dict:
            msg_dict["reasoning_content"] = msg_dict.pop(field_name)
        return msg_dict

    def prepare_request(
        self,
        *,
        model_name: str,
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        enable_streaming: bool,
        provider: ProviderConfig,
        api_key: str | None = None,
        previous_response_id: str | None = None,  # Not used by chat/completions API
    ) -> PreparedRequest:
        field_name = provider.reasoning_field_name
        converted_messages = [
            self._reasoning_to_api(
                msg.model_dump(exclude_none=True, exclude={"message_id"}), field_name
            )
            for msg in messages
        ]

        payload = self.build_payload(
            model_name, converted_messages, temperature, tools, max_tokens, tool_choice
        )

        if enable_streaming:
            payload["stream"] = True
            stream_options = {"include_usage": True}
            if provider.name == "mistral":
                stream_options["stream_tool_calls"] = True
            payload["stream_options"] = stream_options

        headers = self.build_headers(api_key)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        return PreparedRequest(self.endpoint, headers, body)

    def _parse_message(
        self, data: dict[str, Any], field_name: str
    ) -> LLMMessage | None:
        if data.get("choices"):
            choice = data["choices"][0]
            if "message" in choice:
                msg_dict = self._reasoning_from_api(choice["message"], field_name)
                return LLMMessage.model_validate(msg_dict)
            if "delta" in choice:
                msg_dict = self._reasoning_from_api(choice["delta"], field_name)
                return LLMMessage.model_validate(msg_dict)
            raise ValueError("Invalid response data: missing message or delta")

        if "message" in data:
            msg_dict = self._reasoning_from_api(data["message"], field_name)
            return LLMMessage.model_validate(msg_dict)
        if "delta" in data:
            msg_dict = self._reasoning_from_api(data["delta"], field_name)
            return LLMMessage.model_validate(msg_dict)

        return None

    def parse_response(
        self, data: dict[str, Any], provider: ProviderConfig
    ) -> LLMChunk:
        message = self._parse_message(data, provider.reasoning_field_name)
        if message is None:
            message = LLMMessage(role=Role.assistant, content="")

        usage_data = data.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )

        return LLMChunk(message=message, usage=usage)


@register_adapter(RESPONSES_ADAPTERS, "openai")
class OpenAIResponsesAdapter(APIAdapter):
    """Adapter for OpenAI's /responses API endpoint.

    This adapter handles the newer /responses endpoint format which uses
    a different request/response structure compared to /chat/completions.
    """

    endpoint: ClassVar[str] = "/responses"

    def _convert_message_to_input(
        self, msg: LLMMessage, field_name: str
    ) -> dict[str, Any]:
        """Convert an LLMMessage to the responses API input format."""
        if msg.role == Role.tool:
            # Tool results become function_call_output items
            return {
                "type": "function_call_output",
                "call_id": msg.tool_call_id or "",
                "output": msg.content or "",
            }

        if msg.role == Role.assistant and msg.tool_calls:
            # Assistant messages with tool calls need special handling
            # First, add the message content if any
            items: list[dict[str, Any]] = []

            if msg.content:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": msg.content}],
                })

            # Then add function_call items for each tool call
            for tc in msg.tool_calls:
                items.append({
                    "type": "function_call",
                    "call_id": tc.id or "",
                    "name": tc.function.name or "",
                    "arguments": tc.function.arguments or "",
                })

            # Return the first item if only one, otherwise this needs special handling
            # In practice, we'll flatten these in the prepare_request method
            return {"_items": items} if len(items) > 1 else (items[0] if items else {})

        # Regular message (system, user, or assistant without tool calls)
        content: list[dict[str, Any]] = []

        if msg.content:
            content_type = (
                "input_text" if msg.role in (Role.system, Role.user) else "output_text"
            )
            content.append({"type": content_type, "text": msg.content})

        return {"type": "message", "role": str(msg.role), "content": content}

    def build_payload(
        self,
        model_name: str,
        input_items: list[dict[str, Any]],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        store: bool = True,
        previous_response_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_name,
            "input": input_items,
            "temperature": temperature,
            "store": store,
        }

        # For stateful conversations, use previous_response_id to chain responses
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                }
                for tool in tools
            ]

        if tool_choice:
            if isinstance(tool_choice, str):
                # Map standard tool_choice values
                if tool_choice == "required":
                    payload["tool_choice"] = "required"
                elif tool_choice == "none":
                    payload["tool_choice"] = "none"
                else:  # "auto" or "any"
                    payload["tool_choice"] = "auto"
            else:
                # Specific tool selection
                payload["tool_choice"] = {
                    "type": "function",
                    "name": tool_choice.function.name,
                }

        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        return payload

    def build_headers(self, api_key: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def prepare_request(
        self,
        *,
        model_name: str,
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        enable_streaming: bool,
        provider: ProviderConfig,
        api_key: str | None = None,
        previous_response_id: str | None = None,
    ) -> PreparedRequest:
        # Get store setting from provider config
        store = getattr(provider, "responses_api_store", True)

        # Convert messages to input items
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            converted = self._convert_message_to_input(
                msg, provider.reasoning_field_name
            )
            # Handle the case where assistant message with tool calls returns multiple items
            if "_items" in converted:
                input_items.extend(converted["_items"])
            elif converted:
                input_items.append(converted)

        payload = self.build_payload(
            model_name,
            input_items,
            temperature,
            tools,
            max_tokens,
            tool_choice,
            store=store,
            previous_response_id=previous_response_id,
        )

        if enable_streaming:
            payload["stream"] = True

        headers = self.build_headers(api_key)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        return PreparedRequest(self.endpoint, headers, body)

    def parse_response(
        self, data: dict[str, Any], provider: ProviderConfig
    ) -> LLMChunk:
        """Parse a response from the /responses endpoint.

        The responses API uses event-based streaming with different event types.
        This method handles both streaming events and non-streaming responses.
        """
        event_type = data.get("type", "")

        # Handle non-streaming response (full response object) first
        # A full response has "output" key and either no type or a non-event type
        if "output" in data and (not event_type or event_type == "response"):
            return self._parse_full_response(data)

        # Handle streaming events
        if event_type == "response.output_text.delta":
            # Text delta event
            delta_text = data.get("delta", "")
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, content=delta_text), usage=None
            )

        if event_type == "response.function_call_arguments.delta":
            # Function call arguments delta
            delta_args = data.get("delta", "")
            item_id = data.get("item_id", "")
            output_index = data.get("output_index", 0)
            return LLMChunk(
                message=LLMMessage(
                    role=Role.assistant,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=item_id,
                            index=output_index,
                            function=FunctionCall(arguments=delta_args),
                        )
                    ],
                ),
                usage=None,
            )

        if event_type == "response.output_item.added":
            # New output item added - might be a function call
            item = data.get("item", {})
            item_type = item.get("type", "")
            output_index = data.get("output_index", 0)

            if item_type == "function_call":
                # Some servers may include full arguments in output_item.added
                arguments = item.get("arguments", "")
                return LLMChunk(
                    message=LLMMessage(
                        role=Role.assistant,
                        content="",
                        tool_calls=[
                            ToolCall(
                                id=item.get("call_id") or item.get("id", ""),
                                index=output_index,
                                function=FunctionCall(
                                    name=item.get("name", ""), arguments=arguments
                                ),
                            )
                        ],
                    ),
                    usage=None,
                )

            # For message items, return empty chunk (content comes via deltas)
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, content=""), usage=None
            )

        if event_type == "response.function_call_arguments.done":
            # Function call arguments complete - but we've already accumulated via deltas
            # Return empty chunk to avoid double-counting arguments
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, content=""), usage=None
            )

        if event_type in ("response.completed", "response.done"):
            # Response complete - extract usage and response_id if available
            response = data.get("response", {})
            usage_data = response.get("usage", {})
            usage = LLMUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
            )
            response_id = response.get("id") or data.get("id")
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, content=""),
                usage=usage,
                response_id=response_id,
            )

        # Handle non-streaming response (full response object)
        if "output" in data:
            return self._parse_full_response(data)

        # Handle response.created - capture response_id early
        if event_type == "response.created":
            response = data.get("response", {})
            response_id = response.get("id") or data.get("id")
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, content=""),
                usage=None,
                response_id=response_id,
            )

        # Handle response.output_item.done - contains complete item with all data
        # For streaming, we've already accumulated via deltas, so return empty chunk
        # to avoid double-counting arguments
        if event_type == "response.output_item.done":
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, content=""), usage=None
            )

        # Handle other event types (in_progress, etc.) - return empty chunk
        if event_type in (
            "response.in_progress",
            "response.output_text.done",
            "response.content_part.added",
            "response.content_part.done",
        ):
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, content=""), usage=None
            )

        # Unknown event type - return empty chunk
        return LLMChunk(message=LLMMessage(role=Role.assistant, content=""), usage=None)

    def _parse_full_response(self, data: dict[str, Any]) -> LLMChunk:
        """Parse a complete (non-streaming) response from the /responses endpoint."""
        output_items = data.get("output", [])
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for idx, item in enumerate(output_items):
            item_type = item.get("type", "")

            if item_type == "message":
                # Extract text content from message
                for content_item in item.get("content", []):
                    if content_item.get("type") == "output_text":
                        content_parts.append(content_item.get("text", ""))
                    elif content_item.get("type") == "text":
                        content_parts.append(content_item.get("text", ""))

            elif item_type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.get("call_id") or item.get("id", ""),
                        index=idx,
                        function=FunctionCall(
                            name=item.get("name", ""),
                            arguments=item.get("arguments", ""),
                        ),
                    )
                )

        usage_data = data.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
        )

        # Capture response_id for stateful conversations
        response_id = data.get("id")

        return LLMChunk(
            message=LLMMessage(
                role=Role.assistant,
                content="".join(content_parts) if content_parts else None,
                tool_calls=tool_calls if tool_calls else None,
            ),
            usage=usage,
            response_id=response_id,
        )


class GenericBackend:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        provider: ProviderConfig,
        timeout: float = 720.0,
    ) -> None:
        """Initialize the backend.

        Args:
            client: Optional httpx client to use. If not provided, one will be created.
        """
        self._client = client
        self._owns_client = client is None
        self._provider = provider
        self._timeout = timeout

    async def __aenter__(self) -> GenericBackend:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            self._owns_client = True
        return self._client

    async def complete(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
        previous_response_id: str | None = None,
    ) -> LLMChunk:
        api_key = (
            os.getenv(self._provider.api_key_env_var)
            if self._provider.api_key_env_var
            else None
        )

        api_style = getattr(self._provider, "api_style", "openai")
        use_responses_api = getattr(self._provider, "use_responses_api", False)

        # Select appropriate adapter based on configuration
        if use_responses_api:
            adapter = RESPONSES_ADAPTERS.get(
                api_style, RESPONSES_ADAPTERS.get("openai")
            )
        else:
            adapter = BACKEND_ADAPTERS[api_style]

        endpoint, headers, body = adapter.prepare_request(
            model_name=model.name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            enable_streaming=False,
            provider=self._provider,
            api_key=api_key,
            previous_response_id=previous_response_id,
        )

        if extra_headers:
            headers.update(extra_headers)

        url = f"{self._provider.api_base}{endpoint}"

        try:
            res_data, _ = await self._make_request(url, body, headers)
            return adapter.parse_response(res_data, self._provider)

        except httpx.HTTPStatusError as e:
            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                response=e.response,
                headers=e.response.headers,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    async def complete_streaming(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
        previous_response_id: str | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        api_key = (
            os.getenv(self._provider.api_key_env_var)
            if self._provider.api_key_env_var
            else None
        )

        api_style = getattr(self._provider, "api_style", "openai")
        use_responses_api = getattr(self._provider, "use_responses_api", False)

        # Select appropriate adapter based on configuration
        if use_responses_api:
            adapter = RESPONSES_ADAPTERS.get(
                api_style, RESPONSES_ADAPTERS.get("openai")
            )
        else:
            adapter = BACKEND_ADAPTERS[api_style]

        endpoint, headers, body = adapter.prepare_request(
            model_name=model.name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            enable_streaming=True,
            provider=self._provider,
            api_key=api_key,
            previous_response_id=previous_response_id,
        )

        if extra_headers:
            headers.update(extra_headers)

        url = f"{self._provider.api_base}{endpoint}"

        try:
            async for res_data in self._make_streaming_request(url, body, headers):
                yield adapter.parse_response(res_data, self._provider)

        except httpx.HTTPStatusError as e:
            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                response=e.response,
                headers=e.response.headers,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    class HTTPResponse(NamedTuple):
        data: dict[str, Any]
        headers: dict[str, str]

    @async_retry(tries=3)
    async def _make_request(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> HTTPResponse:
        client = self._get_client()
        response = await client.post(url, content=data, headers=headers)
        response.raise_for_status()

        response_headers = dict(response.headers.items())
        response_body = response.json()
        return self.HTTPResponse(response_body, response_headers)

    @async_generator_retry(tries=3)
    async def _make_streaming_request(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> AsyncGenerator[dict[str, Any]]:
        client = self._get_client()
        async with client.stream(
            method="POST", url=url, content=data, headers=headers
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip() == "":
                    continue

                DELIM_CHAR = ":"
                if f"{DELIM_CHAR} " not in line:
                    raise ValueError(
                        f"Stream chunk improperly formatted. "
                        f"Expected `key{DELIM_CHAR} value`, received `{line}`"
                    )
                delim_index = line.find(DELIM_CHAR)
                key = line[0:delim_index]
                value = line[delim_index + 2 :]

                if key != "data":
                    # This might be the case with openrouter, so we just ignore it
                    continue
                if value == "[DONE]":
                    return
                yield json.loads(value.strip())

    async def count_tokens(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        tools: list[AvailableTool] | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> int:
        probe_messages = list(messages)
        if not probe_messages or probe_messages[-1].role != Role.user:
            probe_messages.append(LLMMessage(role=Role.user, content=""))

        result = await self.complete(
            model=model,
            messages=probe_messages,
            temperature=temperature,
            tools=tools,
            max_tokens=16,  # Minimal amount for openrouter with openai models
            tool_choice=tool_choice,
            extra_headers=extra_headers,
        )
        if result.usage is None:
            raise ValueError("Missing usage in non streaming completion")

        return result.usage.prompt_tokens

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None
