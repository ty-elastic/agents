# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-slim 

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser


# Install gcc and other build dependencies.
RUN apt-get update && \
    apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

USER appuser

RUN mkdir -p /home/appuser/.cache
RUN chown -R appuser /home/appuser/.cache

WORKDIR /home/appuser

RUN mkdir deps
COPY --chown=appuser livekit-agents deps/livekit-agents
COPY --chown=appuser livekit-plugins/livekit-plugins-azure deps/livekit-plugins-azure
COPY --chown=appuser livekit-plugins/livekit-plugins-openai deps/livekit-plugins-openai

COPY examples/elk/requirements.txt .
RUN python -m pip install --user --no-cache-dir -r requirements.txt

COPY examples/elk/*.py .

# Run the application.
CMD python elk.py start