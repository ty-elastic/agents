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
import logging
import os
from dataclasses import dataclass
from typing import Optional, AsyncIterable

from livekit import rtc
from livekit.agents import tts, utils

import azure.cognitiveservices.speech as speechsdk

@dataclass
class TTSOptions:
    speech_key: str = None
    speech_region: str = None
    # see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
    voice: str = None

def _create_speech_synthesizer(*, config: TTSOptions, stream : speechsdk.audio.AudioOutputStream) -> speechsdk.SpeechSynthesizer:
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
        super().__init__(streaming_supported=True)

        speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        if not speech_key:
            raise ValueError("AZURE_SPEECH_KEY must be set")
        speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")
        if not speech_region:
            raise ValueError("AZURE_SPEECH_REGION must be set")

        self._config = TTSOptions(
            speech_key=speech_key,
            speech_region=speech_region,
            voice=voice
        )

    def synthesize(
        self,
        *,
        text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        
        # output queue
        results = utils.AsyncIterableQueue()

        async def process():
            nonlocal results

            class PushAudioOutputStreamCallback(speechsdk.audio.PushAudioOutputStreamCallback):
                def __init__(self, push_queue: utils.AsyncIterableQueue):
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

            stream_callback = PushAudioOutputStreamCallback(results)
            push_stream = speechsdk.audio.PushAudioOutputStream(stream_callback)
            speech_synthesizer = _create_speech_synthesizer(config=self._config, stream=push_stream)

            # wait for completion
            result = speech_synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.Canceled:
                print("Speech synthesis canceled: {}".format(result.cancellation_details.reason))

            # Destroys result which is necessary for destroying speech synthesizer
            del result
            # Destroys the synthesizer in order to close the output stream.
            del speech_synthesizer

            results.close()

        asyncio.ensure_future(process())
        return results

    def stream(
        self,
    ) -> tts.SynthesizeStream:
        return SynthesizeStream(self._config)

class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        config: TTSOptions,
    ):
        self._config = config

        self._queue = asyncio.Queue[str]()
        self._event_queue = asyncio.Queue[tts.SynthesisEvent]()
        self._closed = False

        self._main_task = asyncio.create_task(self._run())

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"azure synthesis task failed: {task.exception()}, {task.print_stack()}")
        self._main_task.add_done_callback(log_exception)

        self._text = ""

    def push_text(self, token: str) -> None:
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        if not token or len(token) == 0:
            return

        # TODO: Native word boundary detection may not be good enough for all languages
        # fmt: off
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
        # fmt: on

        self._text += token
        if token[-1] in splitters:
            self._queue.put_nowait(self._text)
            self._text = ""

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
        speech_synthesizer = _create_speech_synthesizer(config=self._config, stream=push_stream)

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
            result = speech_synthesizer.speak_text_async(text).get()
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

    async def flush(self) -> None:
        self._queue.put_nowait(self._text + " ")
        self._text = ""
        await self._queue.join()

    async def aclose(self) -> None:
        await self.flush()
        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def __anext__(self) -> tts.SynthesisEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration
        return await self._event_queue.get()
