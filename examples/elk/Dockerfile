FROM ubuntu:22.04

ARG TARGETPLATFORM

# https://pythonspeed.com/articles/activate-conda-dockerfile/

WORKDIR /home/elk

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential libssl-dev ca-certificates libasound2 wget \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb && \
dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb; \
elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
wget http://ports.ubuntu.com/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_arm64.deb && \
dpkg -i libssl1.1_1.1.1f-1ubuntu2_arm64.deb; \
fi

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
/bin/bash ~/miniconda.sh -b -p /opt/conda; \
elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda.sh && \
/bin/bash ~/miniconda.sh -b -p /opt/conda; \
fi

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create --name elk python=3.12
SHELL ["conda", "run", "-n", "elk", "/bin/bash", "-c"]

RUN conda install -c conda-forge 'ffmpeg<7'

RUN mkdir deps
COPY livekit-agents deps/livekit-agents
COPY livekit-plugins/livekit-plugins-azure deps/livekit-plugins-azure
COPY livekit-plugins/livekit-plugins-openai deps/livekit-plugins-openai

COPY examples/elk/requirements.txt .
RUN python -m pip install --user --no-cache-dir -r requirements.txt

COPY examples/elk/*.py .

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "elk", "python", "elk.py", "start"]
