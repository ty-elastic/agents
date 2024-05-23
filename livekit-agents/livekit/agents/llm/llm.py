from __future__ import annotations

import abc
import enum
from typing import Callable

from attrs import define

from .function_context import FunctionContext


class ChatRole(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@define
class Function:
    arguments: str
    name: str

@define
class ToolCall:
    id: str
    function: Function
    type: str = 'function'

@define
class ChatMessage:
    role: ChatRole
    text: str | None
    id: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = []
    
@define
class ChatContext:
    messages: list[ChatMessage] = []


@define
class ChoiceDelta:
    content: str | None = None
    role: ChatRole | None = None

@define
class Choice:
    delta: ChoiceDelta
    index: int = 0

@define
class ChatChunk:
    choices: list[Choice] = []


class LLM(abc.ABC):
    @abc.abstractmethod
    async def chat(
        self,
        history: ChatContext,
        fnc_ctx: FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
    ) -> "LLMStream": ...


@define
class CalledFunction:
    fnc_name: str
    fnc: Callable
    args: dict
    id: str | None = None
    result: any | None = None

class LLMStream(abc.ABC):
    def __init__(self) -> None:
        # fnc_name, args..
        self._called_functions: list[CalledFunction] = []

    @property
    def called_functions(self) -> list[CalledFunction]:
        """List of called functions from this stream."""
        return self._called_functions
    @called_functions.setter
    def called_functions(self, value):
        self._called_functions = value
        
    @abc.abstractmethod
    def __aiter__(self) -> "LLMStream": ...

    @abc.abstractmethod
    async def __anext__(self) -> ChatChunk: ...

    @abc.abstractmethod
    async def aclose(self, wait: bool = True) -> None: ...
