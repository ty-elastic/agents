import asyncio
import logging
import os
from enum import Enum
from typing import Annotated

from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice_assistant import AssistantContext, VoiceAssistant
from livekit.plugins import deepgram, elevenlabs, openai, silero, azure
from openai import AsyncAzureOpenAI

AZURE_OPENAI_APIVERSION = "2023-12-01-preview"
OPENAI_MODEL = "gpt-4-turbo"

DEBUG_ENABLE = True
if DEBUG_ENABLE:
    logging.basicConfig(level=logging.DEBUG)
    
class Room(Enum):
    BEDROOM = "bedroom"
    LIVING_ROOM = "living room"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    OFFICE = "office"
    
light_state = {}
    
class AssistantFnc(llm.FunctionContext):
    @llm.ai_callable(desc="Turn on/off the lights in a room")
    async def toggle_light(
        self,
        room: Annotated[Room, llm.TypeInfo(desc="The specific room")],
        state: Annotated[bool, llm.TypeInfo(desc="The desired state of the lights; set True for 'on', set False for 'off'")],
    ):
        logging.info("toggle_light %s %s", room, state)
        light_state[room] = state
        ctx = AssistantContext.get_current()
        ctx.store_metadata('changed_room', room)

    @llm.ai_callable(desc="Returns true if a light in the room is on, otherwise false")
    async def get_light_status(
        self,
        room: Annotated[Room, llm.TypeInfo(desc="The specificied room")],
    ) -> bool:
        if room in light_state:
            return room
        else:
            return False

    @llm.ai_callable(desc="User want the assistant to stop/pause speaking")
    def stop_speaking(self):
        pass  # do nothing


async def entrypoint(ctx: JobContext):
    
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not azure_deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT must be set")

        azure_client = AsyncAzureOpenAI(  
            api_key = os.environ["OPENAI_API_KEY"],
            api_version = AZURE_OPENAI_APIVERSION,
            azure_endpoint = azure_endpoint,
            azure_deployment = azure_deployment
        )
        
        gpt = openai.LLM(model=OPENAI_MODEL, client=azure_client)
    else:
        gpt = openai.LLM(model=OPENAI_MODEL)

    initial_ctx = llm.ChatContext(
        messages=[
            llm.ChatMessage(
                role=llm.ChatRole.SYSTEM,
                text="You are a voice assistant created by LiveKit. Your interface with users will be voice. You should use short and concise responses, and avoiding usage of unpronouncable punctuation.",
            )
        ]
    )

    assistant = VoiceAssistant(
        vad=silero.VAD(),
        stt=azure.STT(),
        llm=gpt,
        tts=azure.TTS(),
        fnc_ctx=AssistantFnc(),
        chat_ctx=initial_ctx,
        allow_interruptions=False,
        transcription=False,
    )

    async def _answer_light_toggling(changed_room: str, changed_room_status: bool):
        prompt = "Make a summary of the following actions you did:"
        if changed_room_status:
            prompt += f"\n - You enabled the lights in the following room: {changed_room}"
        else:
            prompt += f"\n - You disabled the lights in the following room: {changed_room}"

        chat_ctx = llm.ChatContext(
            messages=[llm.ChatMessage(role=llm.ChatRole.SYSTEM, text=prompt)]
        )

        stream = await gpt.chat(chat_ctx)
        await assistant.say(stream)

    @assistant.on("agent_speech_interrupted")
    def _agent_speech_interrupted(chat_ctx: llm.ChatContext, msg: llm.ChatMessage):
        msg.text += "... (user interrupted you)"

    @assistant.on("function_calls_finished")
    def _function_calls_done(ctx: AssistantContext):
        logging.info("function_calls_done %s", ctx)

        changed_room = ctx.get_metadata('changed_room', None)
        if changed_room and changed_room in light_state:
            #if there was a change in the lights, summarize it and let the user know
            asyncio.ensure_future(_answer_light_toggling(changed_room, light_state[changed_room]))

    assistant.start(ctx.room)
    await asyncio.sleep(3)
    await assistant.say("Hey, how can I help you today?")


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
