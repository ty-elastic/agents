# ELK - A Talking AI Example Using Elastic AI Assistant and Azure OpenAI

## Running locally

To run the ELK agent locally, install its dependencies:

```bash
pip install -r requirements.txt
```

then start the worker:

```bash
python assistant.py start --api-key=<your livekit api key> --api-secret=<your livekit api secret> --url=<your livekit ws url>
```

This starts the worker and will be listening for new job requests. This worker is configured to listen to the "room" job type (`JT_ROOM`) which means it will get a JobRequest when a new LiveKit room is created.

## How it works

ELK has 4 stages:

- VAD (voice-activity-detection)
- Speech-to-text
- Elastic AI Assistant
- Text-to-speech

Each stage makes use of a plugin. VAD uses `livekit-plugins-silero`, speech-to-text uses `livekit-plugins-openai` and text-to-speech uses `livekit-plugins-openai`.

When a ELK agent starts, it publishes an audio track right away and sends an intro message. It then subscribes to any existing and new audio tracks and sends them into the VAD instance for processing.

When VAD detects that there has been speech, it sends the audio frames containing speech to Deepgram for transcribing. The resulting text is sent to Elastic AI Assistant, which streams a text response. That text response is then sent to elevenlabs to generate audio frames and sent back into the LiveKit room.

## How to deploy

See the `Dockerfile` in the `agents/` dir for reference.
