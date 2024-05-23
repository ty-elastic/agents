import asyncio
import logging
import os
from typing import Annotated
import string
from difflib import SequenceMatcher
from datetime import datetime, timezone

from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    llm,
)
from livekit import agents, rtc
from livekit.protocol import agent
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero, azure
from openai import AsyncAzureOpenAI

from elastic_rag import ElasticRagPlugin, TranscriptionSource

DEBUG_ENABLE = False
if DEBUG_ENABLE:
    logging.basicConfig(level=logging.DEBUG)

AZURE_OPENAI_APIVERSION = "2023-12-01-preview"
OPENAI_MODEL = "gpt-4-turbo"

ASSISTANT_NAME = "elk"
ASSISTANT_KEYWORD = f"hey {ASSISTANT_NAME}"
PROMPT = (f"You are a voice assistant named {ASSISTANT_NAME} whose job it is to help remind participants in a conference as to what was previously said. "
          "Your interface with users will be voice. You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
          "Use only the results returned from your function calls to answer questions, and if it doesn't seem like the results are related to the topic, do not return the results. "
          "Do not ask follow-up questions."
          )

class AssistantFnc(llm.FunctionContext):
    def __init__(self, *, elastic_rag: ElasticRagPlugin, room_id: str):
        super().__init__()
        self._elastic_rag = elastic_rag
        self._room_id = room_id
    
    @llm.ai_callable(desc="Get a list of quotes spoken about a topic")
    async def get_quotes_about_topic(
        self,
        speaker: Annotated[str, llm.TypeInfo(desc="The name of the person who spoke about the topic; use '*' to indicate any speaker")],
        topic: Annotated[str, llm.TypeInfo(desc="The topic the person spoke about")]
    ):
        clauses = await self._elastic_rag.query_transcripts(speaker_name=speaker, query=topic, conference_id=self._room_id)
        return clauses

def detect_keyword(clause: str) -> bool:
    len_of_keyword = len(ASSISTANT_KEYWORD.split(' '))
    clean_clause = clause.translate(str.maketrans('', '', string.punctuation)).lower().split(' ')
    clean_prefix = " ".join(clean_clause[:len_of_keyword])
    s = SequenceMatcher(None,
                        clean_prefix,
                        ASSISTANT_KEYWORD)
    if s.ratio() >= 0.85:
        logging.info(f"elk - recognized assistant request: {clause}")
        return True
    return False

async def create_assistant(*, room: rtc.Room, participant: rtc.Participant, transcript: ElasticRagPlugin, close: asyncio.Event):
    logging.info(f"elk - creating assistant for {participant.name} in room {room.sid}")
                
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
                text=PROMPT
            )
        ]
    )
    
    def on_recv_final_transcript(participant: str, text: str) -> bool:
        keyword_detected = detect_keyword(text)
        if keyword_detected:
            return True
        elif text.strip() != "":
            transcript.push_transcript(
                speaker_name=participant,
                transcription_source=TranscriptionSource.Spoken,
                timestamp=datetime.now(tz=timezone.utc),
                clause=text,
                conference_id=room.sid
            )
            return False

    assistant = VoiceAssistant(
        vad=silero.VAD(),
        stt=azure.STT(grammar=[ASSISTANT_KEYWORD]),
        llm=gpt,
        tts=azure.TTS(),
        fnc_ctx=AssistantFnc(elastic_rag=transcript, room_id=room.sid),
        chat_ctx=initial_ctx,
        on_recv_final_transcript=on_recv_final_transcript,
        debug=DEBUG_ENABLE,
        allow_interruptions=False,
        transcription=False,
    )
    
    announce = True
    assistant.start(room, participant=participant)
    if announce:
        await asyncio.sleep(3)
        await assistant.say(f"Hey {participant.name}, if you need me, just say '{ASSISTANT_KEYWORD}'", allow_interruptions=False)
     
    await close.wait()
    logging.info(f"elk - assistant for {participant.name} in room {room.sid} closing")
    await assistant.aclose(wait=True)


async def entrypoint(ctx: JobContext):
    assistants = {}
    
    transcript = ElasticRagPlugin(
            es_apikey=os.environ.get("ELASTIC_APIKEY"),
            es_endpoint=os.environ.get("ELASTIC_ENDPOINT")
        )
    
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            if participant.identity in assistants:
                return
            close = asyncio.Event()
            assistants[participant.identity] = {
                "task": asyncio.create_task(create_assistant(transcript=transcript, participant=participant, room=ctx.room, close=close)),
                "close": close
            }
    
    @ctx.room.on("disconnected")
    def on_disconnected():
        logging.info(f"elk - room {ctx.room.sid} closed")
        nonlocal assistants
        for assistant in assistants.values():
            assistant['close'].set()
        assistants = {}
        transcript.close()

async def request_fnc(req: JobRequest) -> None:
    logging.info("elk - received request %s", req)
    await req.accept(entrypoint, auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

if __name__ == "__main__":
    ElasticRagPlugin(
            es_apikey=os.environ.get("ELASTIC_APIKEY"),
            es_endpoint=os.environ.get("ELASTIC_ENDPOINT"),
            setup=True,
            start=False
        )
    
    cli.run_app(WorkerOptions(request_fnc, worker_type=agent.JobType.JT_ROOM))
