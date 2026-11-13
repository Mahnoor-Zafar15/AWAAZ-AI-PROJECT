"""Demo Mode scenarios for AWAAZ.

These are PRE-SET scenarios. They let you show the full dashboard in one
click, even if the microphone, internet or models are slow during a pitch.
The scores below are fixed demo values. They are NOT live model output,
and the dashboard says so on screen.

To use your own audio, put files in a folder named `demo_audio` (or keep
them in `sample`). The app looks for a file whose name starts with one of
the `audio_prefixes` of the scenario.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

AUDIO_FOLDERS = ("demo_audio", "sample", "sample_combined")
AUDIO_EXTENSIONS = {".mp3": 0, ".ogg": 1, ".m4a": 2, ".flac": 3, ".wav": 4}


SCENARIOS = {
    "safe": {
        "button": "✅  Safe Call",
        "title": "Safe Call",
        "language": "urdu",
        "fake_score": 0.04,
        "text_score": 0.0,
        "transcript": (
            "السلام علیکم! میں کل شام کو آپ کے گھر آ رہا ہوں۔ "
            "امی سے کہیے گا کہ چائے تیار رکھیں۔"
        ),
        "translation": (
            "Hello! I will come to your house tomorrow evening. "
            "Please tell Mom to have tea ready."
        ),
        "hits": [],
        "audio_prefixes": ["real_normal_ur_", "real_normal_"],
    },
    "clone": {
        "button": "🤖  AI Voice Clone Scam",
        "title": "AI Voice Clone Scam",
        "language": "urdu",
        "fake_score": 0.97,
        "text_score": 92.0,
        "transcript": (
            "ابو، میں بہت مشکل میں ہوں۔ میرا ایکسیڈنٹ ہو گیا ہے۔ "
            "فوراً میرے اکاؤنٹ میں پچاس ہزار روپے بھیج دیں۔ "
            "کسی کو مت بتائیں اور اپنا او ٹی پی بھی جلدی بتا دیں۔"
        ),
        "translation": (
            "Dad, I am in big trouble. I had an accident. Send fifty "
            "thousand rupees to my account right now. Do not tell anyone, "
            "and give me your OTP quickly."
        ),
        "hits": [
            {"category": "Money Request", "weight": 30,
             "keywords": ["پچاس ہزار"]},
            {"category": "OTP / Credentials", "weight": 22,
             "keywords": ["او ٹی پی"]},
            {"category": "Urgency", "weight": 16,
             "keywords": ["فوراً"]},
            {"category": "Secrecy", "weight": 14,
             "keywords": ["کسی کو مت بتائیں"]},
            {"category": "Family Impersonation", "weight": 10,
             "keywords": ["ابو"]},
        ],
        "audio_prefixes": ["ai_sus_ur_17", "ai_sus_"],
    },
    "bank": {
        "button": "🏦  Bank Impersonation (Real Voice)",
        "title": "Bank Impersonation (Real Voice)",
        "language": "urdu",
        "fake_score": 0.06,
        "text_score": 78.0,
        "transcript": (
            "میں بینک سے بات کر رہا ہوں۔ آپ کا اکاؤنٹ فوراً بلاک ہو جائے گا۔ "
            "تصدیق کے لیے آپ کو ابھی جو او ٹی پی کوڈ آیا ہے وہ بتا دیں۔"
        ),
        "translation": (
            "I am speaking from the bank. Your account will be blocked "
            "immediately. To verify, tell me the OTP code you just received."
        ),
        "hits": [
            {"category": "OTP / Credentials", "weight": 28,
             "keywords": ["او ٹی پی"]},
            {"category": "Bank Impersonation", "weight": 22,
             "keywords": ["بینک سے بات"]},
            {"category": "Urgency", "weight": 16,
             "keywords": ["فوراً"]},
            {"category": "Account Threat", "weight": 12,
             "keywords": ["بلاک"]},
        ],
        "audio_prefixes": ["real_sus_ur_3", "real_sus_"],
    },
}


def find_demo_audio(key):
    """Return the path of a sample audio file for this scenario, or None."""
    scenario = SCENARIOS.get(key)
    if not scenario:
        return None

    for folder in AUDIO_FOLDERS:
        folder_path = BASE_DIR / folder
        if not folder_path.is_dir():
            continue

        files = sorted(
            (p for p in folder_path.iterdir()
             if p.suffix.lower() in AUDIO_EXTENSIONS),
            key=lambda p: (AUDIO_EXTENSIONS[p.suffix.lower()], p.name),
        )

        for prefix in scenario["audio_prefixes"]:
            for path in files:
                if path.stem.lower().startswith(prefix.lower()):
                    return str(path)

    return None