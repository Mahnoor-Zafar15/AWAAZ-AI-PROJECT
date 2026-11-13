
"""Awaaz: Scan history and analytics helpers."""

from collections import Counter
from datetime import datetime


def create_record(
    filename,
    language,
    fake_score,
    text_score,
    risk_score,
    verdict,
    transcript,
    hits
):
    """Create one scan-history record."""

    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": str(filename or "Unknown audio"),
        "language": str(language or "Unknown"),
        "fake_score": round(float(fake_score) * 100, 2),
        "text_score": round(float(text_score), 2),
        "risk_score": round(float(risk_score), 2),
        "verdict": str(verdict).upper(),
        "transcript": str(transcript or ""),
        "hits": hits if isinstance(hits, list) else []
    }


def summarize_history(history):
    """Calculate summary statistics from scan history."""

    if not isinstance(history, list) or not history:
        return {
            "total_scans": 0,
            "red": 0,
            "yellow": 0,
            "green": 0,
            "average_risk": 0,
            "average_fake": 0,
            "average_text": 0,
            "patterns": {},
            "timeline": []
        }

    red = sum(
        1 for item in history
        if item.get("verdict") == "RED"
    )

    yellow = sum(
        1 for item in history
        if item.get("verdict") == "YELLOW"
    )

    green = sum(
        1 for item in history
        if item.get("verdict") == "GREEN"
    )

    total = len(history)

    average_risk = sum(
        float(item.get("risk_score", 0))
        for item in history
    ) / total

    average_fake = sum(
        float(item.get("fake_score", 0))
        for item in history
    ) / total

    average_text = sum(
        float(item.get("text_score", 0))
        for item in history
    ) / total

    pattern_counter = Counter()

    for item in history:
        for hit in item.get("hits", []):
            if isinstance(hit, dict):
                category = hit.get("category")

                if category:
                    pattern_counter[category] += 1

    timeline = [
        {
            "time": item.get("time", ""),
            "risk": float(item.get("risk_score", 0)),
            "verdict": item.get("verdict", "GREEN")
        }
        for item in history
    ]

    return {
        "total_scans": total,
        "red": red,
        "yellow": yellow,
        "green": green,
        "average_risk": round(average_risk, 2),
        "average_fake": round(average_fake, 2),
        "average_text": round(average_text, 2),
        "patterns": dict(pattern_counter),
        "timeline": timeline
    }
