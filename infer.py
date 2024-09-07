import base64
import logging
import tempfile
import time

import faster_whisper
import runpod
import torch

from util import download_file

MODEL_NAME = "ivrit-ai/faster-whisper-v2-d3-e3"

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model: {MODEL_NAME} using {device=}")
t0 = time.perf_counter()
model = faster_whisper.WhisperModel(MODEL_NAME, device=device)
t = time.perf_counter() - t0
logger.info(f"Model loaded in {t:.1f} seconds.")

# Maximum data size: 200MB
MAX_PAYLOAD_SIZE = 200 * 1024 * 1024


def mmss(seconds: float):
    m, s = divmod(int(seconds), 60)
    return f"{m:d}:{s:02d}"


def transcribe(job):
    if "input" not in job:
        return {"error": "missing input in job"}

    datatype = job["input"].get("type", None)
    if not datatype:
        return {"error": "datatype field not provided. Should be 'blob' or 'url'."}

    if datatype not in ["blob", "url"]:
        return {
            "error": f"datatype should be 'blob' or 'url', but is {datatype} instead."
        }

    # Get the API key from the job input
    api_key = job["input"].get("api_key", None)

    with tempfile.TemporaryDirectory() as d:
        audio_file = f"{d}/audio.mp3"

        if datatype == "blob":
            mp3_bytes = base64.b64decode(job["input"]["data"])
            open(audio_file, "wb").write(mp3_bytes)
        elif datatype == "url":
            download_url = job["input"]["url"]
            if not download_file(
                download_url,
                MAX_PAYLOAD_SIZE,
                audio_file,
                api_key,
            ):
                return {
                    "error": f"Error downloading data from {download_url}",
                }

        result = transcribe_core(audio_file)
        return {"result": result}


def transcribe_core(audio_file):
    logger.info("Transcribing...")

    segments = []

    segs, dummy = model.transcribe(audio_file, language="he", word_timestamps=True)
    for i, s in enumerate(segs, 1):
        text = s.text or ""
        logger.info(
            f"Segment #{i}: {mmss(s.start)}-{mmss(s.end)} words=~{len(text.split())}\n{s.text!r}"
        )
        words = [
            {
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "score": round(w.probability, 3),
            }
            for w in s.words
        ]

        seg = {
            "id": s.id,
            # "seek": s.seek,
            "start": s.start,
            "end": s.end,
            "text": s.text,
            # "avg_logprob": s.avg_logprob,
            # "compression_ratio": s.compression_ratio,
            # "no_speech_prob": s.no_speech_prob,
            "words": words,
        }

        # pp(seg)
        segments.append(seg)

    return {"segments": segments}


runpod.serverless.start({"handler": transcribe})
