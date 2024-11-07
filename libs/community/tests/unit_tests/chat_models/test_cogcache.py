"""Test OpenAI Chat API wrapper."""

import json
from typing import List

import pytest
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
)

from langchain_community.chat_models.cogcache import (
    ChatCogCache,
    _convert_dict_to_message,
)


def test_cogcache_model_param() -> None:
    test_cases: List[dict] = [{"model": "foo", "api_key": "foo"}]

    for case in test_cases:
        llm = ChatCogCache(**case)
        assert llm.model == "foo", "Model name should be 'foo'"
        if llm.api_key:
            assert llm.api_key.get_secret_value() == "foo", "API key should be 'foo'"


def test__convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = ChatMessage(content="foo", role="system")
    assert result == expected_output


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
        "gpt-4-1106-preview",
        "gpt-35-turbo-0125",
    ],
)
def test_model_name_set_on_chat_result_when_present_in_response(
    model_name: str,
) -> None:
    sample_response_text = f"""
    {{
        "id": "chatcmpl-7ryweq7yc8463fas879t9hdkkdf",
        "object": "chat.completion",
        "created": 1690381189,
        "model": "{model_name}",
        "choices": [
            {{
                "index": 0,
                "finish_reason": "stop",
                "message": {{
                    "role": "assistant",
                    "content": "I'm an AI assistant that can help you."
                }}
            }}
        ],
        "usage": {{
            "completion_tokens": 28,
            "prompt_tokens": 15,
            "total_tokens": 43
        }}
    }}
    """
    # convert sample_response_text to instance of Mapping[str, Any]
    sample_response = json.loads(sample_response_text)
    mock_chat = ChatCogCache(
        api_key="foo",  # type: ignore[arg-type]
        temperature=0,
    )
    chat_result = mock_chat._create_chat_result(sample_response)
    assert (
        chat_result.llm_output is not None
        and chat_result.llm_output["model_name"] == model_name
    )
