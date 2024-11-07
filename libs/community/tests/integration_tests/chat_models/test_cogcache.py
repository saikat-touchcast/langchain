"""Test ChatCogCache."""

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
    LLMResult,
)

from langchain_community.chat_models import ChatCogCache
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.mark.scheduled
def test_chat_cogcache() -> None:
    """Test ChatCogCache wrapper."""
    chat = ChatCogCache(
        model="gpt-4o-2024-08-06",
        temperature=0.7,
        timeout=10.0,
        max_tokens=10,
    )
    message = HumanMessage(content="How to do integration test in langchain?")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_cogcache_model() -> None:
    """Test ChatCogCache wrapper handles model_name."""
    chat = ChatCogCache(model="foo")
    assert chat.model == "foo"
    chat = ChatCogCache(model="bar")  # type: ignore[call-arg]
    assert chat.model == "bar"


def test_chat_cogcache_system_message() -> None:
    """Test ChatCogCache wrapper with system message."""
    chat = ChatCogCache(max_tokens=10, model="gpt-4o-2024-08-06")
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_cogcache_generate() -> None:
    """Test ChatCogCache wrapper with generate."""
    chat = ChatCogCache(
        max_tokens=10,
        n=2,
        model="gpt-4o-2024-08-06",
    )
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output is not None
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
def test_chat_cogcache_multiple_completions() -> None:
    """Test ChatCogCache wrapper with multiple completions."""
    chat = ChatCogCache(
        max_tokens=10,
        n=5,
        model="gpt-4o-2024-08-06",
    )
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)


@pytest.mark.scheduled
def test_chat_cogcache_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatCogCache(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callbacks=callback_manager,
        verbose=True,
        model="gpt-4o-2024-08-06",
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
def test_chat_cogcache_streaming_generation_info() -> None:
    """Test that generation info is preserved when streaming."""

    class _FakeCallback(FakeCallbackHandler):
        saved_things: dict = {}

        def on_llm_end(self, *args: Any, **kwargs: Any) -> Any:
            # Save the generation
            self.saved_things["generation"] = args[0]

    callback = _FakeCallback()
    callback_manager = CallbackManager([callback])
    chat = ChatCogCache(
        max_tokens=2,
        temperature=0,
        callbacks=callback_manager,
        model="gpt-4o-2024-08-06",
    )
    list(chat.stream("hi"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert generation.generations[0][0].text == "Hello!"


def test_chat_cogcache_llm_output_contains_model_name() -> None:
    """Test llm_output contains model name."""
    chat = ChatCogCache(max_tokens=10, model="gpt-4o-2024-08-06")
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model


def test_chat_cogcache_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model name."""
    chat = ChatCogCache(
        max_tokens=10,
        streaming=True,
        model="gpt-4o-2024-08-06",
    )
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model


def test_chat_cogcache_invalid_streaming_params() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        ChatCogCache(
            model="gpt-4o-2024-08-06",
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )


@pytest.mark.scheduled
async def test_async_chat_cogcache() -> None:
    """Test async generation."""
    chat = ChatCogCache(
        max_tokens=10,
        n=2,
        model="gpt-4o-2024-08-06",
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output is not None
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
async def test_async_chat_cogcache_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatCogCache(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callbacks=callback_manager,
        verbose=True,
        model="gpt-4o-2024-08-06",
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test_chat_cogcache_extra_kwargs() -> None:
    """Test extra kwargs to chat cogcache."""
    # Check that foo is saved in extra_kwargs.
    llm = ChatCogCache(
        foo=3,
        max_tokens=10,
        model="gpt-4o-2024-08-06",
    )  # type: ignore[call-arg]
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatCogCache(
        foo=3,
        model_kwargs={"bar": 2},
        model="gpt-4o-2024-08-06",
    )  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatCogCache(
            foo=3,
            model_kwargs={"foo": 2},
            model="gpt-4o-2024-08-06",
        )  # type: ignore[call-arg]

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        ChatCogCache(
            model_kwargs={"temperature": 0.2},
        )

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        ChatCogCache(
            model_kwargs={"model": "gpt-3.5-turbo-instruct"},
            model="gpt-4o-2024-08-06",
        )


@pytest.mark.scheduled
def test_cogcache_streaming() -> None:
    """Test streaming tokens from CogCache."""
    llm = ChatCogCache(max_tokens=10, model="gpt-4o-2024-08-06")

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_cogcache_astream() -> None:
    """Test streaming tokens from CogCache."""
    llm = ChatCogCache(max_tokens=10, model="gpt-4o-2024-08-06")

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_cogcache_abatch() -> None:
    """Test streaming tokens from ChatCogCache."""
    llm = ChatCogCache(max_tokens=10, model="gpt-4o-2024-08-06")

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_cogcache_abatch_tags() -> None:
    """Test batch tokens from ChatCogCache."""
    llm = ChatCogCache(max_tokens=10, model="gpt-4o-2024-08-06")

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_cogcache_batch() -> None:
    """Test batch tokens from ChatCogCache."""
    llm = ChatCogCache(max_tokens=10, model="gpt-4o-2024-08-06")

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_cogcache_ainvoke() -> None:
    """Test invoke tokens from ChatCogCache."""
    llm = ChatCogCache(max_tokens=10, model="gpt-4o-2024-08-06")

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_cogcache_invoke() -> None:
    """Test invoke tokens from ChatCogCache."""
    llm = ChatCogCache(max_tokens=10, model="gpt-4o-2024-08-06")

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


def test_cogcache_invoke_with_response_header() -> None:
    "test invoke with response header"
    llm = ChatCogCache(
        model="gpt-4o-2024-08-06",
        include_response_headers=True,
    )  # type: ignore[call-arg]

    result = llm.invoke("I'm Pickle Rick")
    assert isinstance(result.response_metadata.get("headers"), dict)
