"""Awaaz: AI voice detection and speech transcription."""

import os

# Set this BEFORE importing transformers / huggingface_hub.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import time
from functools import lru_cache

import librosa
import torch
from transformers import pipeline

# Speech-to-text: faster-whisper (CPU, int8).
# Benchmark (12 files): "base" aur "small" ne barabar scam pakde,
# "base" tez tha, isliye base use ho raha hai.
ASR_MODEL = "base"
ASR_COMPUTE_TYPE = "int8"
CPU_THREADS = 4

# ASR sirf shuru ke itne second sunta hai (lambi audio par waqt bachane ke liye).
# None likho to poori audio sunega.
ASR_SECONDS = 30

# 0.0 = Whisper baar baar dobara koshish nahi karta (shor wali audio par
# 30-58 second ki jagah 5-12 second lage, 8 sust files ke test mein).
ASR_TEMPERATURE = 0.0

VOICE_MODEL = (
    "garystafford/wav2vec2-deepfake-voice-detector"
)

SAMPLE_RATE = 16000

# Voice detector ko sirf shuru ke itne second dikhate hain.
# Isse detection tez hoti hai.
VOICE_SECONDS = 5


def get_device():
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def load_voice_model():
    return pipeline(
        "audio-classification",
        model=VOICE_MODEL,
        device=get_device()
    )


@lru_cache(maxsize=1)
def load_asr_model():
    """faster-whisper model sirf ek baar load hota hai (CPU par chalta hai)."""

    from faster_whisper import WhisperModel

    return WhisperModel(
        ASR_MODEL,
        device="cpu",
        compute_type=ASR_COMPUTE_TYPE,
        cpu_threads=CPU_THREADS
    )


# Aakhri load ki hui audio yaad rakhte hain,
# taake voice_check aur asr dobara file na parhein.
_last_audio = {"key": None, "data": None}


def load_audio(audio_path):
    """Audio ko ek hi baar 16kHz mono mein load karta hai."""

    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError("Audio file not found.")

    key = (audio_path, os.path.getmtime(audio_path))

    if _last_audio["key"] == key:
        return _last_audio["data"]

    samples, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    _last_audio["key"] = key
    _last_audio["data"] = samples

    return samples


def voice_check(audio_path, seconds=VOICE_SECONDS):
    """
    Returns AI/fake voice score between 0 and 1.

    seconds: audio ke shuru ke kitne second check karne hain.
             None likho to poori audio check hogi.
    """

    samples = load_audio(audio_path)

    if seconds:
        samples = samples[: int(seconds * SAMPLE_RATE)]

    predictions = load_voice_model()(
        {"raw": samples, "sampling_rate": SAMPLE_RATE},
        top_k=None
    )

    fake_score = 0.0

    for item in predictions:
        label = item["label"].lower()

        if any(word in label for word in (
            "fake", "spoof", "synth",
            "deepfake", "generated"
        )):
            fake_score += float(item["score"])

    return max(0.0, min(1.0, fake_score))


def asr(audio_path, language="urdu"):
    """Converts audio speech into text (faster-whisper)."""

    samples = load_audio(audio_path)

    if ASR_SECONDS:
        samples = samples[: int(ASR_SECONDS * SAMPLE_RATE)]

    if len(samples) == 0:
        return ""

    lang = (
        "ur"
        if str(language).lower() in ("urdu", "اردو", "ur")
        else "en"
    )

    segments, _ = load_asr_model().transcribe(
        samples,
        language=lang,
        beam_size=1,
        condition_on_previous_text=False,
        temperature=ASR_TEMPERATURE
    )

    return " ".join(s.text.strip() for s in segments).strip()


# ---------------------------------------------------------
# SPEED TEST:
#   python detector.py sample_combined\file.ogg
#   python detector.py sample_combined\file.ogg small
# ---------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Use: python detector.py sample_combined\\yourfile.wav")
        sys.exit(1)

    path = sys.argv[1]

    # Agar dusra naam diya ho to wo Whisper model use hoga (base / small)
    if len(sys.argv) >= 3:
        ASR_MODEL = sys.argv[2]

    # File ke naam mein "eng" ho to English, warna Urdu
    lang = "english" if "eng" in os.path.basename(path).lower() else "urdu"

    print("\n--- AWAAZ SPEED TEST ---")
    print("Voice model device:", "GPU" if get_device() == 0 else "CPU")
    print("CPU threads (torch):", torch.get_num_threads())
    print("ASR model: faster-whisper", ASR_MODEL, ASR_COMPUTE_TYPE)
    print("ASR sunta hai pehle (second):", ASR_SECONDS)
    print("Language used:", lang)

    t = time.time()
    samples = load_audio(path)
    t_audio = time.time() - t
    print("Audio length: %.1f sec" % (len(samples) / SAMPLE_RATE))

    t = time.time()
    load_voice_model()
    t_load_voice = time.time() - t

    t = time.time()
    load_asr_model()
    t_load_asr = time.time() - t

    # Warm-up: pehli call ka extra bojh alag nikal dete hain
    voice_check(path, seconds=2)

    t = time.time()
    fake_short = voice_check(path)
    t_voice_short = time.time() - t

    t = time.time()
    text = asr(path, lang)
    t_asr = time.time() - t

    print("\n--- ONE-TIME COSTS (sirf pehli baar) ---")
    print("Voice model load : %.1f sec" % t_load_voice)
    print("Whisper load     : %.1f sec" % t_load_asr)

    print("\n--- HAR SCAN KA TIME ---")
    print("Audio loading                : %.1f sec" % t_audio)
    print("Voice (%d sec audio)          : %.1f sec" % (VOICE_SECONDS, t_voice_short))
    print("Speech-to-text               : %.1f sec" % t_asr)
    print("TOTAL per scan               : %.1f sec" % (t_audio + t_voice_short + t_asr))

    print("\nFake score (%d sec) : %.3f" % (VOICE_SECONDS, fake_short))
    print("\nTranscript:", text)