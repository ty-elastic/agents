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
from livekit.plugins import openai, silero, azure
from openai import AsyncAzureOpenAI

from livekit.plugins.azure import STT

from elastic_transcript import ElasticTranscriptPlugin

from livekit import agents, rtc

class AssistantFnc(llm.FunctionContext):
    def __init__(self, recall):
        self._recall = recall
    
    @llm.ai_callable(desc="Get a summary of what someone said about a topic")
    async def get_summary_of_topic(
        self,
        speaker: Annotated[str, llm.TypeInfo(desc="The name of the person who said something")],
        topic: Annotated[str, llm.TypeInfo(desc="The topic they talked about)]")]
    ):
        # logging.info("toggle_light %s %s", room, status)
        ctx = AssistantContext.get_current()
        # key = "enabled_rooms" if status else "disabled_rooms"
        # li = ctx.get_metadata(key, [])
        # li.append(room)
        # ctx.store_metadata(key, li)
        transcripts = self._recall.query_transcripts(participant_name = speaker, query=topic)
        ctx.store_metadata('transcripts', transcripts)

async def entrypoint(ctx: JobContext):
    tasks = []
    
    elastic_transcript = ElasticTranscriptPlugin(
            es_username=os.environ.get("ELASTIC_TRANSCRIPT_USERNAME"),
            es_password=os.environ.get("ELASTIC_TRANSCRIPT_PASSWORD"),
            es_endpoint=os.environ.get("ELASTIC_TRANSCRIPT_ENDPOINT"),
            es_index_prefix="recall"
        )
    

    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not azure_deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT must be set")

        azure_client = AsyncAzureOpenAI(  
            api_key = os.environ["OPENAI_API_KEY"],
            api_version = "2023-12-01-preview",
            azure_endpoint = azure_endpoint,
            azure_deployment = azure_deployment
        )
        
        gpt = openai.LLM(model="gpt-4-turbo", client=azure_client)
    else:
        gpt = openai.LLM(model="gpt-4-turbo")

    initial_ctx = llm.ChatContext(
        messages=[
            llm.ChatMessage(
                role=llm.ChatRole.SYSTEM,
                text="You are a voice assistant. Your interface with users will be voice. You should use short and concise responses, and avoiding usage of unpronouncable punctuation.",
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
    )
    
    async def process_track(audio_stream: rtc.AudioStream):
        stt = STT()
        stt_stream = stt.stream()
        stt_task = asyncio.create_task(process_stt(stt_stream))
        async for audio_frame_event in audio_stream:
            stt_stream.push_frame(audio_frame_event.frame)
        await stt_task

    async def process_stt(stt_stream: agents.stt.SpeechStream):
        async for stt_event in stt_stream:
            if stt_event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                logging.info("Got transcript: %s", stt_event.alternatives[0].text)

    def on_track_subscribed(track: rtc.Track, *_):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            tasks.append(asyncio.create_task(process_track(rtc.AudioStream(track))))

    ctx.room.on("track_subscribed", on_track_subscribed)

    for participant in ctx.room.participants.values():
        for track_pub in participant.tracks.values():
            # This track is not yet subscribed, when it is subscribed it will
            # call the on_track_subscribed callback
            if track_pub.track is None:
                continue

            tasks.append(
                asyncio.create_task(process_track(rtc.AudioStream(track_pub.track)))
            )

    @assistant.on("agent_speech_interrupted")
    def _agent_speech_interrupted(chat_ctx: llm.ChatContext, msg: llm.ChatMessage):
        msg.text += "... (user interrupted you)"

    @assistant.on("function_calls_finished")
    def _function_calls_done(ctx: AssistantContext):
        logging.info("function_calls_done %s", ctx)
        transcripts = ctx.get_metadata("transcripts", [])
        print(transcripts)

    assistant.start(ctx.room)
    await asyncio.sleep(3)
    await assistant.say("Hey, how can I help you today?")


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
