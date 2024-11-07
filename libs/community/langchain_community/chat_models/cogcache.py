import json
import logging
import os
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    get_pydantic_field_names,
    secret_from_env,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)

## Constants
DEFAULT_API_BASE = "https://proxy-api.cogcache.com/v1"


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    else:
        message_dict = {"role": "assistant", "content": message.content}
    return message_dict


def _convert_dict_to_message(
    _dict: Mapping[str, Any], **kwargs: Any
) -> Union[BaseMessage, None]:
    role = _dict["role"]
    _id = _dict.get("id")

    if role == "user":
        return HumanMessage(content=_dict["content"], id=_id)
    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            id=_id,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    return ChatMessage(content=_dict["content"], role=role, id=_id)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], **kwargs: Any
) -> BaseMessageChunk:
    role = _dict.get("role", "assistant")
    content = _dict.get("content") or ""
    _id = _dict.get("id")
    token_usage = kwargs.get("usage")
    usage_metadata: Optional[UsageMetadata] = (
        UsageMetadata(
            input_tokens=token_usage.get("prompt_tokens", 0),
            output_tokens=token_usage.get("completion_tokens", 0),
            total_tokens=token_usage.get("total_tokens", 0),
        )
        if token_usage
        else None
    )

    additional_kwargs: Dict = {}
    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user":
        return HumanMessageChunk(content=content, id=_id)
    if role == "assistant":
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            usage_metadata=usage_metadata,
            tool_call_chunks=tool_call_chunks,
            id=_id,
        )
    return ChatMessageChunk(content=content, role=role, id=_id)


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


class ChatCogCache(BaseChatModel):
    """`CogCache` chat large language models.

    To use, you should have the ``langchain`` python package installed, and the
    environment variable ``COGCACHE_API_KEY`` set with your API key, or pass
    it as a named parameter(``api_key``) to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatCogCache

            model = ChatCogCache(api_key="my-api-key", model="<model_name>")
    """

    model: str = "gpt-4o-2024-08-06"
    """Model name to use."""

    temperature: float = 0.7
    """What sampling temperature to use."""

    n: int = 1
    """Number of chat completions to generate for each prompt."""

    seed: Optional[int] = None
    """Seed for generation"""

    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate in the completion."""

    stop: Optional[Union[List[str], str]] = Field(default=None, alias="stop_sequences")  # type: ignore[assignment]
    """Default stop sequences."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any additional model parameters to pass along to the CogCache API
    call."""

    default_headers: Dict[str, Any] = Field(default_factory=dict)
    """Holds any default headers to be included in the request."""

    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to CogCache API. Can be float, httpx.Timeout or
        None."""

    api_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("COGCACHE_API_KEY", default=None),
    )
    """CogCache API key"""

    streaming: bool = False
    """Whether to streaming the response to the client. 
    false: if no value is specified or set to false, a non-streaming response is
    returned. "Non-streaming response" means that all responses will be returned at once
    after they are all ready, and the client does not need to concatenate the content.
    true: set to true, partial message deltas will be sent .
    "Streaming response" will provide real-time response of the model to the client, and
    the client needs to assemble the final reply based on the type of message. """

    include_response_headers: bool = False
    """Whether to include response headers in the output message response_metadata."""

    api_base: Optional[str] = Field(default=None)
    """Base URL for CogCache API requests. Leave blank if not using a proxy or service
    emulator."""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": "COGCACHE_API_KEY"}

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.api_base:
            attributes["api_base"] = self.api_base

        return attributes

    @property
    def lc_serializable(self) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    "WARNING! %s is not default parameter. %s was transferred to model_kwargs. Please confirm that %s is what you intended.",  # noqa: E501
                    field_name,
                    field_name,
                    field_name,
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        if not self.api_key:
            raise ValueError(
                "Did not find api_key, please add an environment variable `COGCACHE_API_KEY` which contains it, or pass `api_key` as a named parameter."  # noqa: E501
            )

        self.api_base = (
            self.api_base or os.getenv("COGCACHE_API_BASE") or DEFAULT_API_BASE
        )

        return self

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling CogCache API."""
        exclude_if_none = {
            "model": self.model,
            "stream": self.streaming,
            "seed": self.seed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "stop": self.stop,
        }

        params = {
            **{k: v for k, v in exclude_if_none.items() if v is not None},
            **self.model_kwargs,
        }

        return params

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "CogCache chat"

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        model = self.model
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            model = output.get("model", self.model)
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": model}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined

    def _create_chat_result(
        self,
        response: Mapping[str, Any],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        generations = []
        for c in response.get("choices", []):
            if "message" in c:
                content_dict = c.get("message")

                message = _convert_dict_to_message(content_dict, **response)
                if message:
                    generation_info = generation_info or {}
                    generation_info["finish_reason"] = c.get("finish_reason")
                    if "logprobs" in c:
                        generation_info["logprobs"] = c["logprobs"]
                    generations.append(
                        ChatGeneration(message=message, generation_info=generation_info)
                    )

        llm_output = {
            "token_usage": response.get("usage", {}),
            "model_name": response.get("model", self.model),
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
        parameters = {**self._default_params, **kwargs}

        query = []
        for msg in messages:
            query.append(_convert_message_to_dict(msg))

        if len(query) == 0:
            raise ValueError("At least one input prompt is required")

        headers = parameters.pop("headers", {})
        stream = parameters.pop("stream")
        payload = {"messages": query, "stream": stream, **parameters}

        url = f"{self.api_base}/chat/completions"
        api_key = ""
        if self.api_key:
            api_key = self.api_key.get_secret_value()

        _headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        res = requests.post(
            url=url,
            timeout=self.request_timeout,
            headers={**self.default_headers, **headers, **_headers},
            json=payload,
            stream=stream,
        )
        if res.status_code != 200:
            raise ValueError(
                f"Error from CogCache api response: {res.status_code}: {res.text}"
            )
        return res

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            kwargs["streaming"] = self.streaming
            stream_iter = self._stream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        r = self._chat(messages, **kwargs)
        try:
            res = r.json()
        except json.JSONDecodeError as e:
            logger.error("🔴 Error:%s -> %s", e, r.text)
            raise ValueError(f"Error from CogCache api response: {r.text}") from e

        generation_info: Dict = {}
        if self.include_response_headers:
            generation_info["headers"] = dict(r.headers)
        return self._create_chat_result(res, generation_info)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        res = self._chat(messages, **{**kwargs, **{"stream": True}})
        is_generation_info_needed = True
        for chunk in res.iter_lines():
            chunk = chunk.decode("utf-8").strip("\r\n")
            parts = chunk.split("data:", 1)
            chunk = parts[1] if len(parts) > 1 else parts[0]
            if chunk is None:
                continue

            try:
                response = json.loads(chunk)
            except json.JSONDecodeError:
                continue

            if len(response["choices"]) == 0 and not response.get("usage"):
                continue

            choice = response["choices"][0] if len(response["choices"]) > 0 else {}
            generation_info: Dict = {}
            if self.include_response_headers:
                generation_info["headers"] = dict(res.headers)
            chunk = _convert_delta_to_message_chunk(
                choice.get("message") or choice.get("delta", {}), **response
            )
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
            if model_name := response.get("model"):
                generation_info["model_name"] = model_name
            if system_fingerprint := response.get("system_fingerprint"):
                generation_info["system_fingerprint"] = system_fingerprint

            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs

            cg_chunk = ChatGenerationChunk(
                message=chunk,
                generation_info=generation_info if is_generation_info_needed else {},
            )
            is_generation_info_needed = not kwargs.get("streaming")
            if run_manager:
                run_manager.on_llm_new_token(token=cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "any", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                # 'any' is not natively supported by OpenAI API.
                # We support 'any' since other models use this instead of 'required'.
                if tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if not any(
                    tool_name == tool_choice["function"]["name"]
                    for tool_name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            llm = self.bind_tools([schema], tool_choice=tool_name)
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(response_format={"type": "json_object"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser
