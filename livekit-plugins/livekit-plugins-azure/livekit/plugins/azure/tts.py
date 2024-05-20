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

import contextlib
import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from livekit import rtc
from livekit.agents import tts

import azure.cognitiveservices.speech as speechsdk

@dataclass
class _TTSOptions:
    speech_key: str = None
    speech_region: str = None
    # see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
    voice: str = None

def _create_speech_synthesizer(*, config: _TTSOptions, stream : speechsdk.audio.AudioOutputStream) -> speechsdk.SpeechSynthesizer:
    # Creates an instance of a speech config with specified subscription key and service region.
    speech_config = speechsdk.SpeechConfig(subscription=config.speech_key, region=config.speech_region)
    # add stream output
    stream_config = speechsdk.audio.AudioOutputConfig(stream=stream)
    if config.voice != None:
        # set custom voice, if specified
        speech_config.speech_synthesis_voice_name=config.voice

    return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=stream_config)

class TTS(tts.TTS):
    SAMPLE_RATE: int = 16000
    BITS_PER_SAMPLE: int = 16
    NUM_CHANNELS: int = 1

    def __init__(
        self,
        *,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None,
        voice: Optional[str] = None
    ) -> None:
        super().__init__(streaming_supported=True, sample_rate=TTS.SAMPLE_RATE, num_channels=TTS.NUM_CHANNELS)

        speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        if not speech_key:
            raise ValueError("AZURE_SPEECH_KEY must be set")
        speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")
        if not speech_region:
            raise ValueError("AZURE_SPEECH_REGION must be set")

        self._opts = _TTSOptions(
            speech_key=speech_key,
            speech_region=speech_region,
            voice=voice
        )

    def synthesize(
        self,
        text: str,
    ) -> "ChunkedStream":
        return ChunkedStream(text, self._opts)

    def stream(
        self,
    ) -> "SynthesizeStream":
        return SynthesizeStream(self._opts)

class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, text: str, opts: _TTSOptions
    ) -> None:
        self._opts = opts
        self._text = text
        self._main_task: asyncio.Task | None = None
        self._queue = asyncio.Queue[tts.SynthesizedAudio | None]()
 
    async def _run(self):
        class PushAudioOutputStreamCallback(speechsdk.audio.PushAudioOutputStreamCallback):
            def __init__(self, push_queue: asyncio.Queue[tts.SynthesizedAudio | None]):
                super().__init__()
                self._event_queue = push_queue

            def write(self, audio_buffer: memoryview) -> int:
                # create a rtc frame
                audio_frame = rtc.AudioFrame(
                    data=audio_buffer,
                    sample_rate=TTS.SAMPLE_RATE,
                    num_channels=TTS.NUM_CHANNELS,
                    samples_per_channel=audio_buffer.nbytes // 2,
                )
                # and write to output queue
                self._event_queue.put_nowait(
                    tts.SynthesizedAudio(
                        text="",
                        data=audio_frame
                    )
                )
                return audio_buffer.nbytes

        try:
            stream_callback = PushAudioOutputStreamCallback(self._queue)
            push_stream = speechsdk.audio.PushAudioOutputStream(stream_callback)
            speech_synthesizer = _create_speech_synthesizer(config=self._opts, stream=push_stream)

            # wait for completion
            result = speech_synthesizer.speak_text_async(self._text).get()
            if result.reason == speechsdk.ResultReason.Canceled:
                print("Speech synthesis canceled: {}".format(result.cancellation_details.reason))

            # Destroys result which is necessary for destroying speech synthesizer
            del result
            # Destroys the synthesizer in order to close the output stream.
            del speech_synthesizer

            #results.close()

        except Exception:
            print("failed to synthesize")
        finally:
            self._queue.put_nowait(None)
            
    async def __anext__(self) -> tts.SynthesizedAudio:
        if not self._main_task:
            self._main_task = asyncio.create_task(self._run())

        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration

        return frame

    async def aclose(self) -> None:
        if not self._main_task:
            return

        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task



class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        opts: _TTSOptions
    ):
        self._opts = opts

        self._queue = asyncio.Queue[str | None]()
        self._event_queue = asyncio.Queue[tts.SynthesisEvent | None]()
        self._closed = False
        self._text = ""

        self._main_task = asyncio.create_task(self._run())

    def push_text(self, token: str | None) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        if token is None:
            self._flush_if_needed()
            return

        if len(token) == 0:
            # 11labs marks the EOS with an empty string, avoid users from pushing empty strings
            return

        # TODO: Naive word boundary detection may not be good enough for all languages
        # fmt: off
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
        # fmt: on

        self._text += token

        while True:
            last_split = -1
            for i, c in enumerate(self._text):
                if c in splitters:
                    last_split = i
                    break

            if last_split == -1:
                break

            seg = self._text[: last_split + 1]
            seg = seg.strip() + " "  # 11labs expects a space at the end
            self._queue.put_nowait(seg)
            self._text = self._text[last_split + 1 :]

    async def aclose(self, *, wait: bool = True) -> None:
        self._flush_if_needed()
        self._queue.put_nowait(None)
        self._closed = True

        if not wait:
            self._main_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    def _flush_if_needed(self) -> None:
        seg = self._text.strip()
        if len(seg) > 0:
            self._queue.put_nowait(seg + " ")

        self._text = ""
        
    async def flush(self) -> None:
        self._flush_if_needed()
        await self._queue.join()
        
    async def _run(self) -> None:

        class PushAudioOutputStreamCallback(speechsdk.audio.PushAudioOutputStreamCallback):
            def __init__(self, push_queue: asyncio.Queue[tts.SynthesisEvent]):
                super().__init__()
                self._event_queue = push_queue

            def write(self, audio_buffer: memoryview) -> int:
                # create a rtc frame
                audio_frame = rtc.AudioFrame(
                    data=audio_buffer,
                    sample_rate=TTS.SAMPLE_RATE,
                    num_channels=TTS.NUM_CHANNELS,
                    samples_per_channel=audio_buffer.nbytes // 2,
                )
                # and push it to caller
                self._event_queue.put_nowait(
                    tts.SynthesisEvent(
                        type=tts.SynthesisEventType.AUDIO,
                        audio=tts.SynthesizedAudio(text="", data=audio_frame),
                    )
                )
                return audio_buffer.nbytes

        stream_callback = PushAudioOutputStreamCallback(self._event_queue)
        push_stream = speechsdk.audio.PushAudioOutputStream(stream_callback)
        speech_synthesizer = _create_speech_synthesizer(config=self._opts, stream=push_stream)

        running = False
        # Receives a text from queue and synthesizes it to stream output.
        while True:
            try:
                # temporarily no additional input, but can be restarted anytime
                if self._queue.empty() and running:
                    self._event_queue.put_nowait(
                        tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
                    )
                    running = False
                # wait for new inpurt
                text = await self._queue.get()
                if text is None:
                    break
            except asyncio.CancelledError as e:
                # someone cancelled the wait
                break

            # restarting
            if not running:
                self._event_queue.put_nowait(
                    tts.SynthesisEvent(type=tts.SynthesisEventType.STARTED)
                )
                running = True

            # wait for result
            result = speech_synthesizer.speak_text(text)
            self._queue.task_done()

            if result.reason == speechsdk.ResultReason.Canceled:
                print(f"Speech synthesis canceled: {format(result.cancellation_details.reason)}")
                del result
                break

            # Destroys result which is necessary for destroying speech synthesizer
            del result

        if running:
            self._event_queue.put_nowait(
                tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
            )

        # Destroys the synthesizer in order to close the output stream.
        del speech_synthesizer
        
        self._event_queue.put_nowait(None)

    async def __anext__(self) -> tts.SynthesisEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt
