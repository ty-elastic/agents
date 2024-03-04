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

import asyncio
from datetime import datetime
from enum import Enum
import json
import logging
from typing import AsyncIterable
import os
from datetime import datetime, timezone, timedelta

from livekit import rtc, agents
from livekit.agents.tts import SynthesisEvent, SynthesisEventType
import string

from elastic_transcript import ElasticTranscriptPlugin, TranscriptionType
from livekit.plugins.azure import STT, TTS

INTRO = "Hello, I am Elastic, a friendly assistant powered by Elasticsearch. \
             You can reach me by starting your questions with Hey Elastic."

# convert intro response to a stream
async def intro_text_stream():
    yield INTRO

AgentState = Enum("AgentState", "IDLE, LISTENING, THINKING, SPEAKING")

class ELK:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        elk = ELK(ctx)
        await elk.start()

    def __init__(self, ctx: agents.JobContext):

        self.elastictranscript_plugin = ElasticTranscriptPlugin(
            es_username=os.environ.get("ELASTIC_TRANSCRIPT_USERNAME"),
            es_password=os.environ.get("ELASTIC_TRANSCRIPT_PASSWORD"),
            es_endpoint=os.environ.get("ELASTIC_TRANSCRIPT_ENDPOINT")
        )

        self.stt_plugin = STT(
            keyword_model = "resources/hey-elastic.table"
        )

        self.tts_plugin = TTS(
        )

        self.ctx: agents.JobContext = ctx
        self.chat = rtc.ChatManager(ctx.room)
        self.audio_out = rtc.AudioSource(TTS.SAMPLE_RATE, TTS.NUM_CHANNELS)

        self._sending_audio = False
        self._processing = False
        self._agent_state: AgentState = AgentState.IDLE

        self.chat.on("message_received", self.on_chat_received)
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)

    async def start(self):
        # if you have to perform teardown cleanup, you can listen to the disconnected event
        # self.ctx.room.on("disconnected", your_cleanup_function)

        # publish audio track
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", self.audio_out)
        await self.ctx.room.local_participant.publish_track(track)

        # allow the participant to fully subscribe to the agent's audio track, so it doesn't miss
        # anything in the beginning
        await asyncio.sleep(1)

        await self.process_elasticai_result(intro_text_stream(), type=None)
        self.update_state()

    def on_chat_received(self, message: rtc.ChatMessage):
        if message.deleted:
            return
        
        clause = message.message
        if clause.translate(str.maketrans('', '', string.punctuation)).lower().startswith("hey elastic"):
            start = clause.lower().find("elastic") + len("elastic")
            if clause[start] == ',': start = start+1
            if clause[start] == ' ': start = start+1
            clause = clause[start:]
            print(f"HEY elastic! chat={clause}")
            assistant_stream = self.elastictranscript_plugin.make_assistant_stream(clause)
            self.ctx.create_task(self.process_elasticai_result(assistant_stream, type=TranscriptionType.Chat))
        else:
            self.elastictranscript_plugin.push_transcript(
                participant_name=message.participant.name,
                type=TranscriptionType.Chat,
                timestamp=datetime.now(tz=timezone.utc),
                clause=clause,
                conference_id=self.ctx.room.sid
            )


    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.ctx.create_task(self.process_track(track, participant))

    async def process_track(self, track: rtc.Track, participant: rtc.RemoteParticipant):
        audio_stream = rtc.AudioStream(track)
        stream = self.stt_plugin.stream()
        self.ctx.create_task(self.process_stt_stream(stream, participant))
        async for audio_frame_event in audio_stream:
            if self._agent_state != AgentState.LISTENING:
                continue
            stream.push_frame(audio_frame_event.frame)
        await stream.flush()

    async def process_stt_stream(self, stream, participant: rtc.RemoteParticipant):

        async for event in stream:
            if event.alternatives[0].text == "" or not event.end_of_speech or not event.is_final:
                continue
            
            clause = event.alternatives[0].text

            if clause.translate(str.maketrans('', '', string.punctuation)).lower().startswith("hey elastic"):
                print(f"HEY elastic! STT={clause}")
                start = clause.lower().find("elastic") + len("elastic")
                if clause[start] == ',': start = start+1
                if clause[start] == ' ': start = start+1
                clause = clause[start:]
                assistant_stream = self.elastictranscript_plugin.make_assistant_stream(clause)
                self.ctx.create_task(self.process_elasticai_result(assistant_stream, TranscriptionType.Spoken))
            else:
                print(clause)
                self.elastictranscript_plugin.push_transcript(
                    participant_name=participant.name,
                    timestamp=datetime.now(tz=timezone.utc),
                    clause=clause,
                    type=TranscriptionType.Spoken,
                    conference_id=self.ctx.room.sid
                )

    async def process_elasticai_result(self, text_stream, type : TranscriptionType):
        if type == TranscriptionType.Chat:
            all_text = ""
            async for text in text_stream:
                all_text += text
            await self.chat.send_message(all_text)
        else:
            # ChatGPT is streamed, so we'll flip the state immediately
            self.update_state(processing=True)

            stream = self.tts_plugin.stream()
            # send audio to TTS in parallel
            self.ctx.create_task(self.send_audio_stream(stream))
            all_text = ""
            async for text in text_stream:
                print(f"pushing {text} to TTS")
                stream.push_text(text)
                all_text += text

            self.update_state(processing=False)

            await stream.flush()

    async def send_audio_stream(self, tts_stream: AsyncIterable[SynthesisEvent]):
        async for e in tts_stream:
            if e.type == SynthesisEventType.STARTED:
                self.update_state(sending_audio=True)
            elif e.type == SynthesisEventType.FINISHED:
                self.update_state(sending_audio=False)
            elif e.type == SynthesisEventType.AUDIO:
                await self.audio_out.capture_frame(e.audio.data)
        print("CLOSIN STRWEAM")
        await tts_stream.aclose()
        print("IT IS... CLOSED!")

    def update_state(self, sending_audio: bool = None, processing: bool = None):
        if sending_audio is not None:
            self._sending_audio = sending_audio
        if processing is not None:
            self._processing = processing

        state = AgentState.LISTENING
        if self._sending_audio:
            state = AgentState.SPEAKING
        elif self._processing:
            state = AgentState.THINKING

        self._agent_state = state
        metadata = json.dumps(
            {
                "agent_state": state.name.lower(),
            }
        )
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for Elastic AI Assistant")

        await job_request.accept(
            ELK.create,
            identity="elasticai_agent",
            name="Elastic AI Assistant",
            auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(request_handler=job_request_cb)
    agents.run_app(worker)
