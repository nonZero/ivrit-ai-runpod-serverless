FROM python:3.11.1-buster

WORKDIR /

RUN --mount=type=cache,target=/root/.cache/pip pip install runpod
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.3.1
RUN --mount=type=cache,target=/root/.cache/pip pip install faster-whisper

RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d3-e3")'

ADD util.py .
ADD infer.py .

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib"

CMD [ "python", "-u", "/infer.py" ]

