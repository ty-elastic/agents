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
import contextlib
import time
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, List

from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer, merge_frames

import azure.cognitiveservices.speech as speechsdk

@dataclass
class STTOptions:
    # filename - see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/keyword-recognition-overview
    keyword_model: str
    speech_key: str
    speech_region: str
    # see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=stt
    languages: list[str]
    grammar: list[str]

def _create_speech_recognizer(*, config: STTOptions, stream: speechsdk.audio.AudioInputStream) -> speechsdk.SpeechRecognizer:
    speech_config = speechsdk.SpeechConfig(subscription=config.speech_key, region=config.speech_region)

    # setup for language detection
    auto_detect_source_language_config = None
    if config.languages != []:
        auto_detect_source_language_config = \
            speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=config.languages)

    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config,
                                                   auto_detect_source_language_config=auto_detect_source_language_config)
    
    # add custom phrases to aid in vertical-specific language recognition
    phrase_list_grammar = speechsdk.PhraseListGrammar.from_recognizer(speech_recognizer)
    for phrase in config.grammar:
        phrase_list_grammar.addPhrase(phrase)

    return speech_recognizer

class STT(stt.STT):
    SAMPLE_RATE: int = 16000
    BITS_PER_SAMPLE: int = 16
    NUM_CHANNELS: int = 1
    
    def __init__(
        self,
        *,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None,
        keyword_model: Optional[str] = None,
        languages:  Optional[list[str]] = [],
        grammar: Optional[list[str]] = []
    ):
        super().__init__(streaming_supported=True)

        speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        if not speech_key:
            raise ValueError("AZURE_SPEECH_KEY must be set")
        speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")
        if not speech_region:
            raise ValueError("AZURE_SPEECH_REGION must be set")
        
        self._config = STTOptions(
            keyword_model=keyword_model,
            speech_key=speech_key,
            speech_region=speech_region,
            languages=languages,
            grammar=grammar
        )

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
    ) -> stt.SpeechEvent:
        
        class PullInputStreamCallback(speechsdk.audio.PullAudioInputStreamCallback):
            def __init__(self, buffer: AudioBuffer):
                super().__init__()
                self._buffer = merge_frames(buffer)
                self._buffer_pos = 0

            def read(self, buffer: memoryview) -> int:
                # EOF
                if self._buffer.data.nbytes - self._buffer_pos <= 0:
                    return 0
                # read a chunk out of the buffer
                bytes_to_copy = min(buffer.nbytes, self._buffer.data.nbytes - self._buffer_pos)
                buffer[:bytes_to_copy] = self._buffer.data.tobytes()[self._buffer_pos:self._buffer_pos+bytes_to_copy]
                self._buffer_pos += bytes_to_copy
                return bytes_to_copy

            def close(self):
                pass

        # align STT format to input format
        sample_width = buffer.data.nbytes / buffer.samples_per_channel
        wave_format = speechsdk.audio.AudioStreamFormat(samples_per_second=buffer.sample_rate, bits_per_sample=int(8*sample_width),
                                                        channels=buffer.num_channels)
        callback = PullInputStreamCallback(buffer)
        stream = speechsdk.audio.PullAudioInputStream(callback, wave_format)

        speech_recognizer = _create_speech_recognizer(config=self._config, stream=stream)

        done = False

        def stop_cb(evt):
            # fired when EOF reached on pull input stream or cancellation
            nonlocal done
            done = True

        # gather all recognized clauses
        text = ""
        language = ""
        def recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            nonlocal text
            text += evt.result.text
            nonlocal language
            auto_detect_source_language_result = speechsdk.AutoDetectSourceLanguageResult(evt.result)
            language = auto_detect_source_language_result.language

        # Connect callbacks to the events fired by the speech recognizer
        speech_recognizer.recognized.connect(recognized_cb)
        #speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
        #speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
        #speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
        #speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
        # stop continuous recognition on either session stopped or canceled events
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)

        # Start continuous speech recognition
        speech_recognizer.start_continuous_recognition()
        # this will stop when pull stream is EOF
        while not done:
            time.sleep(.5)
        speech_recognizer.stop_continuous_recognition()

        # return gathered STT data
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=language,
                    start_time=0,
                    end_time=0,
                    confidence=1,
                    text=text
                )
            ],
        )

    def stream(
        self
    ) -> "SpeechStream":
        return SpeechStream(
            self._config
        )

class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        config: STTOptions
    ) -> None:
        super().__init__()

        self._config = config
        
        self._queue = asyncio.Queue()
        self._event_queue = asyncio.Queue[stt.SpeechEvent]()
        self._closed = False
        self._speech_recognizer = None
        self._speaking = False

        self._main_task = asyncio.create_task(self._run())

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"azure recognition task failed: {task.exception()}, {task.print_stack()}")
        self._main_task.add_done_callback(log_exception)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")
        self._queue.put_nowait(frame)

    async def aclose(self, wait: bool = True) -> None:
        self._closed = True
        if not wait:
            self._main_task.cancel()

        self._queue.put_nowait(None)
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    def _start_stream_recognition(self, data : rtc.AudioFrame) -> speechsdk.audio.PushAudioInputStream:
        # align format to first frame
        sample_width = data.data.nbytes / data.samples_per_channel
        wave_format = speechsdk.audio.AudioStreamFormat(samples_per_second=data.sample_rate,
                                                        bits_per_sample=int(8*sample_width),
                                                        channels=data.num_channels)
        stream = speechsdk.audio.PushAudioInputStream(stream_format=wave_format)
        self._speech_recognizer = _create_speech_recognizer(config=self._config, stream=stream)

        def cancelled_cb(evt: speechsdk.SessionEventArgs):
            #print('CANCELLED: {}'.format(evt))
            self._queue.put_nowait(None)

        def recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            #print('RECOGNIZED: {}'.format(evt))
            if evt.result.reason == speechsdk.ResultReason.RecognizedKeyword or evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                auto_detect_source_language_result = speechsdk.AutoDetectSourceLanguageResult(evt.result)
                se = stt.SpeechEvent(
                    type = stt.SpeechEventType.END_OF_SPEECH,
                    alternatives=[
                        stt.SpeechData(
                            language=auto_detect_source_language_result.language,
                            start_time=0,
                            end_time=0,
                            confidence=1,
                            text=evt.result.text
                        )
                    ],
                )
                self._event_queue.put_nowait(se)

        def started_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            #print('STARTED: {}'.format(evt))
            self._speaking = True
            start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
            self._event_queue.put_nowait(start_event)

        def stopped_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            #print('STOPPED: {}'.format(evt))
            self._speaking = False

        # Connect callbacks to the events fired by the speech recognizer
        self._speech_recognizer.recognized.connect(recognized_cb)
        #self._speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
        self._speech_recognizer.session_started.connect(started_cb)
        self._speech_recognizer.session_stopped.connect(stopped_cb)
        self._speech_recognizer.canceled.connect(cancelled_cb)

        # Start continuous speech recognition
        if self._config.keyword_model is not None:
            model = speechsdk.KeywordRecognitionModel(self._config.keyword_model)
            print("model loaded")
            self._speech_recognizer.start_keyword_recognition(model)
        else:
            self._speech_recognizer.start_continuous_recognition()
            
        return stream

    async def _run(self) -> None:
        stream = None
        while not self._closed:
            # get a frame from the input queue
            data = await self._queue.get()

            # check if audio or cntrl
            if isinstance(data, rtc.AudioFrame):
                frame = data
                # resample to required rate for azure STT
                if frame.sample_rate != STT.SAMPLE_RATE or frame.num_channels != STT.NUM_CHANNELS:
                    frame = data.remix_and_resample(STT.SAMPLE_RATE, STT.NUM_CHANNELS)

                # if we haven't started yet, init speech recognition
                if stream is None:
                    stream = self._start_stream_recognition(frame)
                stream.write(frame.data.tobytes())
            elif data is None:
                break

            self._queue.task_done()

        if self._config.keyword_model is not None:
            self._speech_recognizer.stop_keyword_recognition()
        else:
            self._speech_recognizer.stop_continuous_recognition()
        
        self._event_queue.put_nowait(None)

    async def __anext__(self) -> stt.SpeechEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt
