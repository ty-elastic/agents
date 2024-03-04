# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import asyncio
import openai
from dataclasses import dataclass
from typing import AsyncIterable, List, Optional
from enum import Enum
import json

ChatGPTMessageRole = Enum("MessageRole", ["system", "user", "assistant", "function"])

@dataclass
class ChatGPTMessage:
    role: ChatGPTMessageRole
    content: str
    id: Optional[str]
    name: Optional[str]

    def __init__(self, role: ChatGPTMessageRole, content: str, id: Optional[str] = None, name: Optional[str] = None):
        self.role = role
        self.content = content
        self.id = id
        self.name = name

    def to_api(self):
        msg = {"role": self.role.name, "content": self.content}
        if self.id is not None:
            msg['tool_call_id'] = self.id
        if self.name is not None:
            msg['name'] = self.name
        return msg


class ChatGPTPlugin:
    """OpenAI ChatGPT Plugin"""

    def __init__(self, prompt: str, message_capacity: int, model: str, tools, call_function):
        """
        Args:
            prompt (str): First 'system' message sent to the chat that prompts the assistant
            message_capacity (int): Maximum number of messages to send to the chat
            model (str): Which model to use (i.e. 'gpt-3.5-turbo')
        """
        self._model = model
        
        self._prompt = prompt
        self._message_capacity = message_capacity
        self._messages: List[ChatGPTMessage] = []
        self._producing_response = False
        self._needs_interrupt = False
        self._tools = tools
        self._call_function = call_function

        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            self._client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            if not azure_deployment:
                raise ValueError("AZURE_OPENAI_DEPLOYMENT must be set")

            self._client = openai.AsyncAzureOpenAI(  
                api_key = os.environ["OPENAI_API_KEY"],  
                api_version = "2023-12-01-preview",
                azure_endpoint = azure_endpoint,
                azure_deployment = azure_deployment
            )

    def interrupt(self):
        """Interrupt a currently streaming response (if there is one)"""
        if self._producing_response:
            self._needs_interrupt = True

    async def aclose(self):
        pass

    async def send_system_prompt(self) -> AsyncIterable[str]:
        """Send the system prompt to the chat and generate a streamed response

        Returns:
            AsyncIterable[str]: Streamed ChatGPT response
        """
        async for text in self.add_message(None):
            yield text

    async def add_message(
        self, message: Optional[ChatGPTMessage]
    ) -> AsyncIterable[object]:
        """Add a message to the chat and generate a streamed response

        Args:
            message (ChatGPTMessage): The message to add

        Returns:
            AsyncIterable[str]: Streamed ChatGPT response
        """

        print("ADD MSG")

        if message is not None:
            self._messages.append(message)
        if len(self._messages) > self._message_capacity:
            self._messages.pop(0)

        async for text in self._generate_text_streamed(self._model):
            yield text

    async def _generate_text_streamed(self, model: str) -> AsyncIterable[object]:
        prompt_message = ChatGPTMessage(
            role=ChatGPTMessageRole.system, content=self._prompt
        )
        try:
            print('here')
            chat_messages = [m.to_api() for m in self._messages]
            print(chat_messages)
            chat_stream = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model,
                    n=1,
                    stream=True,
                    tools=self._tools,
                    messages=[prompt_message.to_api()] + chat_messages,
                ),
                10,
            )
        except TimeoutError:
            yield "Sorry, I'm taking too long to respond. Please try again later."
            return

        self._producing_response = True
        complete_response = ""
        complete_function = None

        async def anext_util(aiter):
            async for item in aiter:
                return item

            return None

        while True:
            try:
                chunk = await asyncio.wait_for(anext_util(chat_stream), 5)
            except TimeoutError:
                break
            except asyncio.CancelledError:
                self._producing_response = False
                self._needs_interrupt = False
                break

            if chunk is None:
                break
            #print(chunk)
            if len(chunk.choices) == 0:
                continue

# ChatCompletionChunk(id='chatcmpl-8wc8wuzmbKKisskjFelMh6psznKhf',
#                     choices=[Choice(delta=ChoiceDelta(content=None, 
#                                                       function_call=None, 
#                                                       role=None, 
#                                                       tool_calls=[ChoiceDeltaToolCall(index=0, 
#                                                                                       id=None, 
#                                                                                       function=ChoiceDeltaToolCallFunction(arguments='}', 
#                                                                                                                            name=None), 
#                                                                                                                            type=None)
#                                                                     ]), 
#                        finish_reason=None, 
#                        index=0, 
#                        logprobs=None, 
#                         content_filter_results={}
#  )]

            if chunk.choices[0].delta.tool_calls is not None and len(chunk.choices[0].delta.tool_calls) > 0:
                if chunk.choices[0].delta.tool_calls[0].function.name != None:
                    complete_function = {'arguments': ""}
                    complete_function['name'] = chunk.choices[0].delta.tool_calls[0].function.name
                    complete_function['id'] = chunk.choices[0].delta.tool_calls[0].id
                else:
                    complete_function['arguments'] += chunk.choices[0].delta.tool_calls[0].function.arguments
                #print(chunk.choices[0])
            elif chunk.choices[0].finish_reason == 'tool_calls':
                print(complete_function)
                self._messages.append(
                    ChatGPTMessage(role=ChatGPTMessageRole.assistant, content=json.dumps(complete_response))
                )

                complete_function['arguments'] = json.loads( complete_function['arguments'])
                
                
                #self._call_function(complete_function)
                yield complete_function
                #return
            else:
                content = chunk.choices[0].delta.content
                if content is not None:
                    complete_response += content
                    yield content

            if self._needs_interrupt:
                self._needs_interrupt = False
                logging.info("ChatGPT interrupted")
                break

        self._messages.append(
            ChatGPTMessage(role=ChatGPTMessageRole.assistant, content=complete_response)
        )
        self._producing_response = False
        #print("DONEDONE")
