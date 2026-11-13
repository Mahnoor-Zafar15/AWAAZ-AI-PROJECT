---
title: AWAAZ AI - Voice Scam Shield
emoji: "🔐"
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.12"
app_file: app.py
startup_duration_timeout: 45m
models:
  - garystafford/wav2vec2-deepfake-voice-detector
short_description: AI voice authenticity and scam-content screening prototype.
---
# AWAAZ AI — Voice Scam Shield

A two-layer security shield that checks a voice for **AI cloning** and checks
what is being said for **scam behaviour**, then fuses both into one
**Threat Risk Score**.

## Architecture

```
 audio (upload or microphone)
        |
        +--> Layer 1: acoustic deepfake detection (wav2vec2 classifier)
        |        -> AI-voice probability, Real / AI / Inconclusive label
        |
        +--> Layer 2: speech-to-text (Whisper) -> scam analysis
        |        (rule-based Urdu + English: money requests, OTP,
        |         urgency, secrecy, impersonation)
        |        -> scam text score + highlighted findings
        |
        +--> Fusion: Threat Risk Score (0-100) and final verdict
                 SAFE / SUSPICIOUS / HIGH-RISK SCAM DETECTED
```

| File | Job |
|---|---|
| `app.py` | Dashboard, scan pipeline, latency timing, waveform/spectrogram, Demo Mode |
| `detector.py` | Layer 1 voice model and Whisper speech-to-text |
| `scam_engine.py` | Layer 2 scam rules, risk fusion, final level |
| `analytics.py` | Session scan history and statistics |
| `demo_data.py` | Preset Demo Mode scenarios |
| `assets/style.css` | Cyber dark theme |

## Tech stack

Python, Gradio, PyTorch, Hugging Face Transformers, faster-whisper,
librosa, matplotlib.

## Setup (Windows PowerShell)

```powershell
cd Desktop\AWAAZ
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

The first run downloads the models, so use the internet once. For an offline
pitch, run `$env:HF_HUB_OFFLINE="1"` before `python app.py`.

Google Colab: upload all project files, run `pip install -r requirements.txt`,
then `!python app.py`. A public HTTPS link is created, which the microphone needs.

Optional settings: `AWAAZ_WARMUP=0` (skip model warm-up), `AWAAZ_PORT`,
`AWAAZ_HOST`, `AWAAZ_SHARE=1`.

## Demo Mode

Three one-click scenarios (Safe Call, AI Voice Clone Scam, Bank
Impersonation with a real voice) fill the whole dashboard instantly.
**Their scores and transcripts are preset, not live model output**, and the
screen says so. Sample audio is picked from `demo_audio/`, `sample/` or
`sample_combined/`. Press ANALYZE AUDIO afterwards to run the live pipeline
on the loaded file. To use your own measured numbers, edit `demo_data.py`.

## Limitations (please read)

- The detectors are experimental. Results are indicators, not proof.
- The evaluation dataset is small, and AI and real files may differ in
  source format and length. See the "Test results" tab for the honest numbers.
- Layer 1 is tuned on ElevenLabs-style voices. Other voice-cloning tools may
  not be detected.
- Scam rules are keyword based, so they can miss new scam wording or
  heavily mis-transcribed speech.
- Always verify a suspicious request through a trusted channel.