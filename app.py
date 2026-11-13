"""Awaaz AI - Voice Scam Shield

Two-layer pipeline:
  Layer 1  acoustic deepfake detection      -> detector.voice_check
  Layer 2  speech-to-text + scam analysis   -> detector.asr + scam_engine.text_check
  Fusion   one Threat Risk Score + verdict  -> scam_engine.combined_risk / final_level
"""

# Set this BEFORE importing transformers / huggingface_hub.
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import base64
import html
import io
import re
import sys
import tempfile
import time
import wave
from pathlib import Path

import gradio as gr

from detector import voice_check, asr

from scam_engine import (
    text_check,
    combined_risk,
    final_level,
    security_assessment
)

from analytics import create_record, summarize_history

from demo_data import SCENARIOS, find_demo_audio


# =========================================================
# PATHS, CSS AND THEME
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "assets" / "style.css"

# Only plot the first seconds of audio so memory stays small.
MAX_PLOT_SECONDS = 30

EXTRA_CSS = r"""
:root {
    --aw-card: rgba(11, 20, 34, 0.88);
    --aw-line: rgba(0, 229, 195, 0.20);
    --aw-text: #e6f7f4;
    --aw-muted: #8ea3b5;
    --aw-teal: #00e5c3;
}

.aw-card {
    background: var(--aw-card);
    border: 1px solid var(--aw-line);
    border-radius: 16px;
    padding: 20px 22px;
    color: var(--aw-text);
    box-sizing: border-box;
    height: 100%;
}
.aw-card h3 { margin: 2px 0 12px; font-size: 1.15rem; color: var(--aw-text); }
.aw-eyebrow {
    font-size: 0.72rem; letter-spacing: 0.16em; font-weight: 700;
    text-transform: uppercase; color: var(--aw-teal);
}
.aw-muted { color: var(--aw-muted); font-size: 0.88rem; margin: 10px 0 0; }
.aw-meta { color: var(--aw-muted); font-size: 0.82rem; margin-top: 12px; }

/* ---------- verdict banner ---------- */
.aw-banner {
    display: flex; align-items: center; gap: 20px;
    padding: 22px 26px; margin: 6px 0 10px; border-radius: 18px;
    border: 1px solid var(--result-color);
    background: #0d1620;
    background: linear-gradient(135deg,
        color-mix(in srgb, var(--result-color) 22%, #0a1018), #0a1018 70%);
    box-shadow: 0 0 28px color-mix(in srgb, var(--result-color) 28%, transparent);
    color: var(--aw-text);
}
.aw-banner-icon { font-size: 2.8rem; line-height: 1; }
.aw-banner-text { flex: 1; min-width: 0; }
.aw-banner-title {
    font-size: clamp(1.3rem, 3.4vw, 2.3rem); font-weight: 800;
    letter-spacing: 0.02em; line-height: 1.15; margin: 4px 0;
    color: var(--result-color);
}
.aw-banner-sub { color: var(--aw-muted); font-size: 0.98rem; }
.aw-banner-risk { text-align: center; min-width: 96px; }
.aw-banner-risk strong {
    display: block; font-size: 2.6rem; line-height: 1; color: var(--result-color);
}
.aw-banner-risk small { color: var(--aw-muted); font-size: 0.68rem; letter-spacing: 0.1em; }
.aw-banner-red { animation: aw-pulse 1.8s ease-in-out infinite; }
@keyframes aw-pulse {
    0%, 100% { box-shadow: 0 0 20px color-mix(in srgb, var(--result-color) 25%, transparent); }
    50% { box-shadow: 0 0 46px color-mix(in srgb, var(--result-color) 55%, transparent); }
}
@media (prefers-reduced-motion: reduce) { .aw-banner-red { animation: none; } }

.aw-demo-tag {
    display: inline-block; margin-top: 10px; padding: 3px 10px;
    border-radius: 999px; font-size: 0.72rem; letter-spacing: 0.1em;
    font-weight: 700; color: #1a1300; background: #ffc857;
}

/* ---------- risk gauge ---------- */
.aw-risk-row { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; margin-top: 10px; }
.aw-gauge {
    position: relative; width: 150px; height: 150px; border-radius: 50%;
    display: grid; place-items: center; flex: none;
    background: conic-gradient(var(--result-color) calc(var(--pct) * 1%),
                               rgba(255, 255, 255, 0.08) 0);
}
.aw-gauge::before {
    content: ""; position: absolute; inset: 13px; border-radius: 50%;
    background: #0b1422;
}
.aw-gauge-num {
    position: relative; font-size: 2.5rem; font-weight: 800; color: var(--result-color);
}
.aw-risk-bars { flex: 1; min-width: 200px; }

.aw-bar-label {
    display: flex; justify-content: space-between; margin: 10px 0 5px;
    font-size: 0.86rem; color: var(--aw-text);
}
.aw-bar { height: 9px; border-radius: 999px; background: rgba(255, 255, 255, 0.08); overflow: hidden; }
.aw-bar-fill { height: 100%; border-radius: 999px; }
.aw-fill-voice { background: linear-gradient(90deg, #00b8d4, #00e5c3); }
.aw-fill-text { background: linear-gradient(90deg, #ffc857, #ff5268); }

/* ---------- layer cards ---------- */
.aw-big { font-size: 1.55rem; font-weight: 800; letter-spacing: 0.02em; margin: 4px 0 2px; }
.aw-sub { color: var(--aw-muted); font-size: 0.92rem; margin-bottom: 12px; }
.aw-dual { display: flex; height: 12px; border-radius: 999px; overflow: hidden; background: rgba(255, 255, 255, 0.08); }
.aw-dual-real { background: #00d6a3; }
.aw-dual-ai { background: #ff5268; }
.aw-dual-legend { display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.8rem; color: var(--aw-muted); }

.aw-chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
.aw-chip {
    padding: 4px 11px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    color: #ffd7dd; background: rgba(255, 82, 104, 0.16);
    border: 1px solid rgba(255, 82, 104, 0.4);
}
.aw-transcript {
    margin: 10px 0; padding: 14px 16px; border-radius: 12px;
    background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08);
    font-size: 1.2rem; line-height: 2; color: var(--aw-text);
}
.aw-translation { color: var(--aw-muted); font-size: 0.88rem; font-style: italic; margin: 4px 0 10px; }
.aw-mark {
    background: rgba(255, 82, 104, 0.28); color: #ffe3e7;
    border-radius: 5px; padding: 0 4px;
}
.aw-finding {
    display: flex; justify-content: space-between; align-items: center; gap: 12px;
    padding: 8px 0; border-top: 1px solid rgba(255, 255, 255, 0.07);
}
.aw-finding b { color: var(--aw-text); }
.aw-finding p { margin: 2px 0 0; color: var(--aw-muted); font-size: 0.85rem; }
.aw-finding span { color: #ff8a9a; font-weight: 700; }

/* ---------- performance + plot ---------- */
.aw-perf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-top: 10px; }
.aw-perf-item {
    padding: 12px 14px; border-radius: 12px; text-align: center;
    background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(0, 229, 195, 0.14);
}
.aw-perf-item span { display: block; font-size: 0.7rem; letter-spacing: 0.1em; color: var(--aw-muted); text-transform: uppercase; }
.aw-perf-item strong { display: block; margin-top: 4px; font-size: 1.25rem; color: var(--aw-teal); }
.aw-plot-img { width: 100%; border-radius: 12px; margin-top: 10px; display: block; }

/* ---------- demo panel ---------- */
.aw-demo-panel {
    padding: 14px 18px; border-radius: 14px; margin-bottom: 8px;
    border: 1px dashed rgba(255, 200, 87, 0.55); background: rgba(255, 200, 87, 0.06);
    color: var(--aw-text);
}
.aw-demo-panel .aw-eyebrow { color: #ffc857; }
.aw-demo-panel p { margin: 6px 0 0; color: var(--aw-muted); font-size: 0.9rem; }

@media (max-width: 640px) {
    .aw-banner { flex-direction: column; text-align: center; }
}
"""

BASE_CSS = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
CSS = BASE_CSS + "\n" + EXTRA_CSS

THEME = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="cyan",
    neutral_hue="slate"
)


BACKGROUND_JS = r"""() => {
    const canvasId = "awaaz-cyber-canvas";
    if (document.getElementById(canvasId)) return;

    const canvas = document.createElement("canvas");
    canvas.id = canvasId;
    canvas.setAttribute("aria-hidden", "true");
    document.body.prepend(canvas);

    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    let width = 0;
    let height = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const pointer = { x: -1000, y: -1000, active: false };
    const count = window.innerWidth < 700 ? 28 : 48;

    const particles = Array.from({ length: count }, () => ({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        vx: (Math.random() - 0.5) * 0.22,
        vy: (Math.random() - 0.5) * 0.22,
        r: Math.random() * 1.5 + 0.7,
        phase: Math.random() * Math.PI * 2
    }));

    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        dpr = Math.min(window.devicePixelRatio || 1, 1.5);
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = width + "px";
        canvas.style.height = height + "px";
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function onMove(event) {
        pointer.x = event.clientX;
        pointer.y = event.clientY;
        pointer.active = true;
    }

    window.addEventListener("resize", resize, { passive: true });
    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointerleave", () => {
        pointer.active = false;
    });

    resize();

    function frame(time) {
        ctx.clearRect(0, 0, width, height);

        if (pointer.active) {
            const glow = ctx.createRadialGradient(
                pointer.x, pointer.y, 0,
                pointer.x, pointer.y, 230
            );

            glow.addColorStop(0, "rgba(0, 229, 195, 0.105)");
            glow.addColorStop(0.42, "rgba(0, 170, 180, 0.035)");
            glow.addColorStop(1, "rgba(0, 170, 180, 0)");

            ctx.fillStyle = glow;
            ctx.fillRect(
                pointer.x - 230,
                pointer.y - 230,
                460,
                460
            );

            ctx.beginPath();
            ctx.arc(pointer.x, pointer.y, 3, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(111, 255, 232, 0.9)";
            ctx.shadowColor = "rgba(0, 229, 195, 0.9)";
            ctx.shadowBlur = 16;
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];

            p.x += p.vx;
            p.y += p.vy;

            if (p.x < -10) p.x = width + 10;
            if (p.x > width + 10) p.x = -10;
            if (p.y < -10) p.y = height + 10;
            if (p.y > height + 10) p.y = -10;

            if (pointer.active) {
                const dx = p.x - pointer.x;
                const dy = p.y - pointer.y;
                const distance = Math.hypot(dx, dy);

                if (distance < 125 && distance > 0.1) {
                    const force = (125 - distance) / 125 * 0.018;
                    p.x += dx * force;
                    p.y += dy * force;
                }
            }

            for (let j = i + 1; j < particles.length; j++) {
                const q = particles[j];
                const dx = p.x - q.x;
                const dy = p.y - q.y;
                const distance = Math.hypot(dx, dy);

                if (distance < 118) {
                    const alpha = (1 - distance / 118) * 0.18;

                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(q.x, q.y);
                    ctx.strokeStyle = `rgba(0, 213, 190, ${alpha})`;
                    ctx.lineWidth = 0.7;
                    ctx.stroke();
                }
            }

            if (pointer.active) {
                const distance = Math.hypot(
                    p.x - pointer.x,
                    p.y - pointer.y
                );

                if (distance < 155) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(pointer.x, pointer.y);
                    ctx.strokeStyle =
                        `rgba(75, 255, 220, ${(1 - distance / 155) * 0.28})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }

            const pulse = 0.65 + Math.sin(time * 0.001 + p.phase) * 0.22;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(90, 238, 215, ${pulse})`;
            ctx.shadowColor = "rgba(0, 229, 195, 0.55)";
            ctx.shadowBlur = 7;
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        requestAnimationFrame(frame);
    }

    if (!window.matchMedia(
        "(prefers-reduced-motion: reduce)"
    ).matches) {
        requestAnimationFrame(frame);
    }
}"""


# =========================================================
# SMALL HELPERS
# =========================================================

def safe_text(value):
    return html.escape(str(value or ""))



def recommendation(level, fake_score, text_score):
    """Display independent AI voice and scam-content alerts."""

    assessment = security_assessment(fake_score, text_score)

    ai_alert = assessment["ai_voice_alert"]
    scam_warning = assessment["scam_warning"]
    verification_advice = assessment["verification_advice"]

    alerts = ""

    if ai_alert:
        alerts += """
        <div class="alert-item">
            <b>AI VOICE ALERT</b>
            <p>AI-generated or cloned-voice indicators detected.
            Independently verify the speaker's identity.</p>
        </div>
        """

    if scam_warning:
        alerts += """
        <div class="alert-item">
            <b>SCAM BEHAVIOUR WARNING</b>
            <p>Suspicious conversation patterns detected.
            Verify the request through an independent trusted channel.</p>
        </div>
        """

    elif verification_advice:
        alerts += """
        <div class="alert-item">
            <b>REQUEST VERIFICATION</b>
            <p>A potentially sensitive request was detected.
            Verify the request independently before taking action.</p>
        </div>
        """

    if not ai_alert and not verification_advice:
        alerts = """
        <div class="alert-item">
            <b>NO STRONG INDICATORS</b>
            <p>No strong AI-voice or scam-content indicators detected.
            This does not guarantee that the audio is safe.</p>
        </div>
        """

    if ai_alert and scam_warning:
        heading = "Multiple Security Warnings"
    elif ai_alert:
        heading = "Voice Authenticity Warning"
    elif scam_warning:
        heading = "Suspicious Conversation Warning"
    elif verification_advice:
        heading = "Caution Recommended"
    else:
        heading = "No Strong Threat Indicators"

    return f"""
    <div>
        <h3>{heading}</h3>
        {alerts}
        <p class="muted-text">
            AI voice indicators do not independently prove fraud.
            Voice authenticity and conversation behaviour are
            assessed separately.
        </p>
    </div>
    """



def verdict_color(level):
    return {
        "RED": "#ff5268",
        "YELLOW": "#ffc857",
        "GREEN": "#00d6a3"
    }.get(level, "#ffc857")


# =========================================================
# HOME AND HOW IT WORKS SCREENS
# =========================================================

HOME = """
<section class="hero-section">
    <div class="hero-eyebrow">
        AI POWERED VOICE SECURITY
    </div>

    <h1 class="hero-title">
        AWAAZ <span>AI</span>
    </h1>

    <p class="hero-subtitle">AI Voice Scam Shield</p>

    <p class="hero-description">
        An intelligent voice analysis system designed to identify
        potentially AI-generated voices and suspicious scam-related
        speech in Urdu and English audio.
    </p>

    <div class="hero-status">
        <span class="status-dot"></span>
        VOICE INTELLIGENCE SYSTEM
    </div>
</section>

<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon"></div>
        <h3>Voice Analysis</h3>
        <p>Examines audio for possible AI-generated voice patterns.</p>
    </div>

    <div class="feature-card">
        <div class="feature-icon"></div>
        <h3>Scam Detection</h3>
        <p>Checks transcripts for suspicious language and scam patterns.</p>
    </div>

    <div class="feature-card">
        <div class="feature-icon"></div>
        <h3>Risk Intelligence</h3>
        <p>Combines voice and text signals into a risk assessment.</p>
    </div>
</div>

<div class="info-panel">
    <div class="section-eyebrow">SYSTEM WORKFLOW</div>
    <h3>How Awaaz works</h3>
    <p>
        Upload audio <span>→</span> Analyze voice <span>→</span>
        Convert speech into text <span>→</span>
        Check scam patterns <span>→</span> View risk assessment.
    </p>
</div>
"""


# =========================================================
# HOW IT WORKS SCREEN
# =========================================================

HOW = """
<div class="content-panel">
    <div class="section-eyebrow">BEHIND THE SYSTEM</div>
    <h2>How the system works</h2>

    <div class="process-step">
        <span>01</span>
        <div>
            <h3>Audio Input</h3>
            <p>The user uploads an audio recording for analysis.</p>
        </div>
    </div>

    <div class="process-step">
        <span>02</span>
        <div>
            <h3>AI Voice Detection</h3>
            <p>A pretrained audio-classification model estimates
            whether the voice may be real or AI-generated.</p>
        </div>
    </div>

    <div class="process-step">
        <span>03</span>
        <div>
            <h3>Speech Recognition</h3>
            <p>Whisper converts spoken words into text.
            The application supports Urdu and English selection.</p>
        </div>
    </div>

    <div class="process-step">
        <span>04</span>
        <div>
            <h3>Scam Pattern Analysis</h3>
            <p>The transcript is checked for suspicious patterns such as
            money requests, urgency, secrecy and impersonation.</p>
        </div>
    </div>

    <div class="process-step">
        <span>05</span>
        <div>
            <h3>Final Risk Assessment</h3>
            <p>Voice and text signals are combined to produce a GREEN,
            YELLOW or RED risk indication.</p>
        </div>
    </div>

    <div class="notice-panel">
        <b>Important:</b>
        This is an experimental detection system. Results are indicators,
        not proof that a voice or message is genuine or fraudulent.
        Always verify suspicious requests independently.
    </div>
</div>
"""


# =========================================================
# RECOMMENDATION CARD
# =========================================================

def recommendation_card(level, fake_score, text_score):
    return f"""
    <div class="result-card recommendation-card">
        <div class="section-eyebrow">SAFETY GUIDANCE</div>
        <h3>Recommended Action</h3>
        <div class="recommendation-text">
            {recommendation(level, fake_score, text_score)}
        </div>
    </div>
    """


# =========================================================
# ANALYTICS SCREEN
# =========================================================

def analytics_panel(history):
    data = summarize_history(history)

    if data["total_scans"] == 0:
        return """
        <div class="empty-state">
            <div class="empty-icon">📡</div>
            <h3>No Analytics Yet</h3>
            <p>Complete your first audio scan to see analytics here.</p>
        </div>
        """

    total = data["total_scans"]

    pattern_html = ""

    if data["patterns"]:
        for name, count in sorted(
            data["patterns"].items(),
            key=lambda item: item[1],
            reverse=True
        ):
            pattern_html += f"""
            <div class="pattern-row">
                <span>{safe_text(name)}</span>
                <b>{count}</b>
            </div>
            """
    else:
        pattern_html = '<p class="muted-text">No scam patterns recorded.</p>'

    return f"""
    <div class="analytics-grid">
        <div class="stat-card">
            <span>TOTAL SCANS</span>
            <strong>{total}</strong>
        </div>
        <div class="stat-card stat-red">
            <span>RED ALERTS</span>
            <strong>{data["red"]}</strong>
        </div>
        <div class="stat-card stat-yellow">
            <span>CAUTION</span>
            <strong>{data["yellow"]}</strong>
        </div>
        <div class="stat-card stat-green">
            <span>LOW RISK</span>
            <strong>{data["green"]}</strong>
        </div>
    </div>

    <div class="analytics-detail-grid">
        <div class="result-card">
            <h3>Average Scores</h3>
            <div class="pattern-row">
                <span>Risk Score</span>
                <b>{data["average_risk"]}/100</b>
            </div>
            <div class="pattern-row">
                <span>AI Voice Score</span>
                <b>{data["average_fake"]}%</b>
            </div>
            <div class="pattern-row">
                <span>Scam Text Score</span>
                <b>{data["average_text"]}%</b>
            </div>
        </div>

        <div class="result-card">
            <h3>Detected Patterns</h3>
            {pattern_html}
        </div>
    </div>
    """


# =========================================================
# THREAT CENTER — ALERTS AND SCAN GRAPH
# =========================================================

def threat_center_panel(history):
    data = summarize_history(history)

    if not history:
        return """
        <div class="threat-empty">
            <div class="pulse-orb"></div>
            <div class="section-eyebrow">THREAT MONITOR</div>
            <h2>Monitoring Standby</h2>
            <p>
                The Threat Center will display scan alerts and risk trends
                after you analyze your first audio recording.
            </p>
        </div>
        """

    recent = history[-10:]
    bars = ""

    for index, item in enumerate(recent, start=1):
        risk = max(0, min(100, float(item.get("risk_score", 0))))
        level = str(item.get("verdict", "GREEN")).upper()
        color = verdict_color(level)

        bars += f"""
        <div class="graph-column">
            <div class="graph-value">{risk:.0f}</div>
            <div class="graph-bar-wrap">
                <div class="graph-bar"
                     style="height:{max(4, risk)}%;background:{color}">
                </div>
            </div>
            <div class="graph-label">{index}</div>
        </div>
        """

    latest = history[-1]
    latest_level = str(latest.get("verdict", "GREEN")).upper()
    latest_color = verdict_color(latest_level)

    latest_file = safe_text(latest.get("filename", "Unknown audio"))
    latest_risk = float(latest.get("risk_score", 0))

    alert_text = {
        "RED": "High-risk indicators detected. Verify the request independently.",
        "YELLOW": "Caution indicators detected. Verify the speaker and request.",
        "GREEN": "No strong scam indicators detected. Continue to use caution."
    }.get(latest_level, "Review the latest scan result.")

    return f"""
    <div class="threat-header">
        <div>
            <div class="section-eyebrow">LIVE SESSION MONITOR</div>
            <h2>Threat Intelligence Center</h2>
        </div>
        <div class="monitor-status">
            <span class="status-dot"></span> SESSION ACTIVE
        </div>
    </div>

    <div class="analytics-grid">
        <div class="stat-card">
            <span>SCANS MONITORED</span>
            <strong>{data["total_scans"]}</strong>
        </div>
        <div class="stat-card stat-red">
            <span>HIGH RISK</span>
            <strong>{data["red"]}</strong>
        </div>
        <div class="stat-card stat-yellow">
            <span>CAUTION</span>
            <strong>{data["yellow"]}</strong>
        </div>
        <div class="stat-card stat-green">
            <span>LOW RISK</span>
            <strong>{data["green"]}</strong>
        </div>
    </div>

    <div class="alert-panel" style="--result-color:{latest_color}">
        <div class="alert-pulse"></div>
        <div>
            <div class="section-eyebrow">LATEST SCAN ALERT</div>
            <h3>{safe_text(latest_level)} — {latest_risk:.0f}/100</h3>
            <p>{alert_text}</p>
            <small>File: {latest_file}</small>
        </div>
    </div>

    <div class="result-card graph-card">
        <div class="section-eyebrow">RECENT SCAN HISTORY</div>
        <h3>Risk Trend — Latest {len(recent)} Scans</h3>
        <p class="muted-text">
            Each bar represents one scan. Higher bars indicate higher risk.
        </p>
        <div class="risk-graph">
            {bars}
        </div>
        <div class="graph-legend">
            <span><i class="legend-red"></i> High risk</span>
            <span><i class="legend-yellow"></i> Caution</span>
            <span><i class="legend-green"></i> Low risk</span>
        </div>
    </div>
    """


# =========================================================
# SCAN HISTORY TABLE
# =========================================================

def history_rows(history):
    rows = []

    for item in reversed(history[-20:]):
        rows.append([
            item.get("time", ""),
            item.get("filename", ""),
            item.get("verdict", ""),
            item.get("risk_score", 0)
        ])

    return rows


def refresh_analytics(history):
    history = history or []

    return (
        analytics_panel(history),
        history_rows(history),
        threat_center_panel(history)
    )


# =========================================================
# FORMAT HELPERS
# =========================================================

def fmt_time(ms):
    if ms is None:
        return "—"
    ms = float(ms)
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


BANNERS = {
    "RED": (
        "⛔",
        "HIGH-RISK SCAM DETECTED",
        "Do not send money or share OTPs or passwords. "
        "Verify through a trusted channel."
    ),
    "YELLOW": (
        "⚠️",
        "SUSPICIOUS — VERIFY BEFORE YOU ACT",
        "Some risk indicators were found. "
        "Confirm the caller's identity independently."
    ),
    "GREEN": (
        "✅",
        "SAFE — NO STRONG THREAT INDICATORS",
        "No strong AI-voice or scam signals were found. "
        "No detector is perfect, so stay alert."
    ),
}


def idle_banner():
    return """
    <div class="aw-banner aw-banner-idle" style="--result-color:#00e5c3"
         role="status">
        <div class="aw-banner-icon">🛡️</div>
        <div class="aw-banner-text">
            <div class="aw-eyebrow">SYSTEM READY</div>
            <div class="aw-banner-title">AWAITING AUDIO</div>
            <div class="aw-banner-sub">
                Upload a file, record with your microphone, or run a
                demo scenario above.
            </div>
        </div>
    </div>
    """


def verdict_banner(level, risk, demo_title=None):
    level = str(level).upper()
    key = level if level in BANNERS else "YELLOW"
    icon, title, sub = BANNERS[key]
    color = verdict_color(key)

    tag = ""
    if demo_title:
        tag = (
            '<div><span class="aw-demo-tag">DEMO SCENARIO · '
            f'{safe_text(demo_title)}</span></div>'
        )

    return f"""
    <div class="aw-banner aw-banner-{key.lower()}"
         style="--result-color:{color}" role="status">
        <div class="aw-banner-icon">{icon}</div>
        <div class="aw-banner-text">
            <div class="aw-eyebrow">FINAL VERDICT · {safe_text(key)}</div>
            <div class="aw-banner-title">{title}</div>
            <div class="aw-banner-sub">{sub}</div>
            {tag}
        </div>
        <div class="aw-banner-risk">
            <strong>{float(risk):.0f}</strong>
            <small>/100<br>THREAT RISK</small>
        </div>
    </div>
    """


# =========================================================
# RESULT CARDS
# =========================================================

def risk_card(risk, fake, text_score, level):
    color = verdict_color(str(level).upper())
    pct = max(0.0, min(100.0, float(risk)))
    voice_pct = max(0.0, min(100.0, float(fake) * 100))
    text_pct = max(0.0, min(100.0, float(text_score)))

    return f"""
    <div class="aw-card" style="--result-color:{color}">
        <div class="aw-eyebrow">UNIFIED THREAT RISK SCORE</div>
        <div class="aw-risk-row">
            <div class="aw-gauge" style="--pct:{pct:.1f}">
                <span class="aw-gauge-num">{pct:.0f}</span>
            </div>
            <div class="aw-risk-bars">
                <div class="aw-bar-label">
                    <span>Layer 1 · AI voice score</span>
                    <b>{voice_pct:.1f}%</b>
                </div>
                <div class="aw-bar">
                    <div class="aw-bar-fill aw-fill-voice"
                         style="width:{voice_pct:.1f}%"></div>
                </div>
                <div class="aw-bar-label">
                    <span>Layer 2 · Scam text score</span>
                    <b>{text_pct:.1f}%</b>
                </div>
                <div class="aw-bar">
                    <div class="aw-bar-fill aw-fill-text"
                         style="width:{text_pct:.1f}%"></div>
                </div>
                <p class="aw-muted">
                    The voice layer and the speech layer are combined into
                    one 0–100 score.
                </p>
            </div>
        </div>
    </div>
    """


def voice_verdict(fake):
    """Turn the AI-voice probability into a readable label.

    The 0.35 to 0.65 zone is shown as INCONCLUSIVE on purpose: a detector
    that is unsure should say so instead of guessing.
    """
    if fake >= 0.65:
        return "AI-GENERATED VOICE", "#ff5268", fake
    if fake <= 0.35:
        return "REAL HUMAN VOICE", "#00d6a3", 1 - fake
    return "INCONCLUSIVE", "#ffc857", max(fake, 1 - fake)


def layer1_card(fake, timings, demo):
    label, color, confidence = voice_verdict(fake)
    ai_pct = max(0.0, min(100.0, fake * 100))
    real_pct = 100.0 - ai_pct

    if demo:
        meta = "Preset demo value (no live inference)."
    else:
        meta = f"Layer 1 latency: {fmt_time(timings['voice_ms'])}"

    return f"""
    <div class="aw-card">
        <div class="aw-eyebrow">LAYER 1 · ACOUSTIC ANALYSIS</div>
        <h3>Deepfake Voice Detection</h3>
        <div class="aw-big" style="color:{color}">{label}</div>
        <div class="aw-sub">Confidence {confidence * 100:.1f}%</div>
        <div class="aw-dual">
            <div class="aw-dual-real" style="width:{real_pct:.1f}%"></div>
            <div class="aw-dual-ai" style="width:{ai_pct:.1f}%"></div>
        </div>
        <div class="aw-dual-legend">
            <span>Real {real_pct:.1f}%</span>
            <span>AI-generated {ai_pct:.1f}%</span>
        </div>
        <div class="aw-meta">{meta}</div>
    </div>
    """


def text_verdict(text_score):
    if text_score >= 60:
        return "HIGH SCAM LANGUAGE", "#ff5268"
    if text_score >= 30:
        return "SUSPICIOUS LANGUAGE", "#ffc857"
    return "NO SCAM LANGUAGE", "#00d6a3"


def highlight_transcript(transcript, hits):
    """Escape the transcript and mark the words that triggered a finding."""
    text = html.escape(str(transcript or ""))

    if not text:
        return "No speech detected."

    entity_parts = {"amp", "quot", "lt", "gt", "x27"}
    words = set()

    for hit in hits or []:
        keywords = hit.get("keywords", []) or []

        if isinstance(keywords, str):
            keywords = [keywords]

        for keyword in keywords:
            keyword = html.escape(str(keyword).strip())

            if keyword and keyword not in entity_parts:
                words.add(keyword)

    if not words:
        return text

    pattern = re.compile(
        "|".join(
            re.escape(word)
            for word in sorted(words, key=len, reverse=True)
        ),
        re.IGNORECASE
    )

    return pattern.sub(
        lambda match: f'<mark class="aw-mark">{match.group(0)}</mark>',
        text
    )


def layer2_card(transcript, hits, text_score, language, timings, demo):
    label, color = text_verdict(text_score)

    chips = "".join(
        f'<span class="aw-chip">{safe_text(hit.get("category", "Unknown"))}'
        f' +{safe_text(hit.get("weight", 0))}</span>'
        for hit in (hits or [])
    )

    if chips:
        chips = f'<div class="aw-chip-row">{chips}</div>'

    findings = ""

    for hit in hits or []:
        keywords = hit.get("keywords", []) or []

        if isinstance(keywords, str):
            keywords = [keywords]

        findings += f"""
        <div class="aw-finding">
            <div>
                <b>{safe_text(hit.get("category", "Unknown"))}</b>
                <p>{safe_text(", ".join(str(k) for k in keywords))}</p>
            </div>
            <span>+{safe_text(hit.get("weight", 0))}</span>
        </div>
        """

    if not findings:
        findings = (
            '<p class="aw-muted">'
            'No matching scam patterns were detected.</p>'
        )

    translation = ""
    if demo and demo.get("translation"):
        translation = (
            '<div class="aw-translation">English meaning: '
            f'{safe_text(demo["translation"])}</div>'
        )

    if demo:
        meta = "Preset demo transcript (no live speech recognition)."
    else:
        meta = (
            f"Speech-to-text ({safe_text(str(language).title())}): "
            f"{fmt_time(timings['asr_ms'])} · "
            f"Scam analysis: {fmt_time(timings['text_ms'])}"
        )

    return f"""
    <div class="aw-card">
        <div class="aw-eyebrow">LAYER 2 · SEMANTIC ANALYSIS</div>
        <h3>Scam Language Detection</h3>
        <div class="aw-big" style="color:{color}">{label}</div>
        <div class="aw-sub">Scam text score {text_score:.1f}%</div>
        {chips}
        <div class="aw-transcript" dir="auto">
            {highlight_transcript(transcript, hits)}
        </div>
        {translation}
        {findings}
        <div class="aw-meta">{meta}</div>
    </div>
    """


def performance_card(timings, duration, demo):
    if demo:
        return """
        <div class="aw-card">
            <div class="aw-eyebrow">SYSTEM PERFORMANCE</div>
            <h3>Pipeline Latency</h3>
            <p class="aw-muted">
                This is a preset demo scenario, so no live inference ran.
                Real timings appear after you press ANALYZE AUDIO.
            </p>
        </div>
        """

    speed = ""

    if duration and duration > 0:
        factor = (timings["total_ms"] / 1000.0) / duration
        speed = f"""
        <div class="aw-perf-item">
            <span>Speed</span>
            <strong>{factor:.2f}× audio</strong>
        </div>
        """

    return f"""
    <div class="aw-card">
        <div class="aw-eyebrow">SYSTEM PERFORMANCE</div>
        <h3>Pipeline Latency</h3>
        <div class="aw-perf-grid">
            <div class="aw-perf-item">
                <span>Layer 1 · Voice</span>
                <strong>{fmt_time(timings["voice_ms"])}</strong>
            </div>
            <div class="aw-perf-item">
                <span>Speech-to-text</span>
                <strong>{fmt_time(timings["asr_ms"])}</strong>
            </div>
            <div class="aw-perf-item">
                <span>Layer 2 · Scam</span>
                <strong>{fmt_time(timings["text_ms"])}</strong>
            </div>
            <div class="aw-perf-item">
                <span>Total</span>
                <strong>{fmt_time(timings["total_ms"])}</strong>
            </div>
            {speed}
        </div>
        <p class="aw-muted">
            Speed below 1.0× means the analysis finished faster than the
            audio length.
        </p>
    </div>
    """


# =========================================================
# WAVEFORM + SPECTROGRAM
# =========================================================

def plot_message(text):
    return f"""
    <div class="aw-card">
        <div class="aw-eyebrow">SIGNAL VISUALIZATION</div>
        <h3>Waveform &amp; Spectrogram</h3>
        <p class="aw-muted">{safe_text(text)}</p>
    </div>
    """


def make_audio_visual(audio_path):
    """Return (html, duration_seconds) with a waveform and a mel-spectrogram.

    Only the first MAX_PLOT_SECONDS are loaded, so long files cannot use
    a lot of memory. Any problem returns a friendly message instead.
    """
    if not audio_path:
        return plot_message("No audio file to draw."), None

    try:
        import numpy as np
        import librosa
        from matplotlib.figure import Figure

        samples, rate = librosa.load(
            audio_path, sr=16000, mono=True, duration=MAX_PLOT_SECONDS
        )

        if samples.size == 0:
            return plot_message("The audio file is empty."), None

        shown = samples.size / rate

        try:
            duration = float(librosa.get_duration(path=audio_path))
        except Exception:
            duration = shown

        figure = Figure(figsize=(10, 4.4), dpi=80, facecolor="#0b1422")
        wave_axis, spec_axis = figure.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 1.5]}
        )

        # Waveform (min/max envelope keeps the drawing light and clear)
        blocks = 1000
        if samples.size >= blocks * 2:
            size = samples.size // blocks
            trimmed = samples[: size * blocks].reshape(blocks, size)
            times = (np.arange(blocks) + 0.5) * size / rate
            wave_axis.fill_between(
                times, trimmed.min(axis=1), trimmed.max(axis=1),
                color="#00e5c3", linewidth=0
            )
        else:
            wave_axis.plot(
                np.arange(samples.size) / rate, samples,
                color="#00e5c3", linewidth=0.8
            )

        wave_axis.set_xlim(0, shown)
        wave_axis.set_title("Waveform", loc="left", color="#8ea3b5", fontsize=10)

        # Mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=samples, sr=rate, n_fft=1024, hop_length=256,
            n_mels=64, fmax=8000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        spec_axis.imshow(
            mel_db, origin="lower", aspect="auto", cmap="magma",
            extent=[0, shown, 0, 8000]
        )
        spec_axis.set_title(
            "Mel-spectrogram", loc="left", color="#8ea3b5", fontsize=10
        )
        spec_axis.set_xlabel("Time (seconds)", color="#8ea3b5", fontsize=9)
        spec_axis.set_ylabel("Hz", color="#8ea3b5", fontsize=9)

        for axis in (wave_axis, spec_axis):
            axis.set_facecolor("#0b1422")
            axis.tick_params(colors="#8ea3b5", labelsize=8)
            for spine in axis.spines.values():
                spine.set_color("#1f3042")

        wave_axis.set_yticks([])
        figure.tight_layout()

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", facecolor=figure.get_facecolor())
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        note = ""
        if duration > MAX_PLOT_SECONDS:
            note = (
                f'<p class="aw-muted">Showing the first {shown:.0f} seconds '
                f'of {duration:.0f}.</p>'
            )

        return f"""
        <div class="aw-card">
            <div class="aw-eyebrow">SIGNAL VISUALIZATION</div>
            <h3>Waveform &amp; Spectrogram</h3>
            <img class="aw-plot-img" alt="Waveform and spectrogram"
                 src="data:image/png;base64,{encoded}">
            {note}
        </div>
        """, duration

    except Exception as error:
        print("AWAAZ VISUAL ERROR:", repr(error))
        return plot_message(
            "The waveform could not be drawn for this file."
        ), None


# =========================================================
# RESULT ASSEMBLY
# =========================================================

def render_result(level, risk, fake, text_score, transcript, hits,
                  timings, duration, plot_html, language, history,
                  demo=None):
    """Build the 11 values that every scan and demo returns to the UI."""
    return (
        verdict_banner(level, risk, demo["title"] if demo else None),
        risk_card(risk, fake, text_score, level),
        layer1_card(fake, timings, demo),
        layer2_card(transcript, hits, text_score, language, timings, demo),
        plot_html,
        performance_card(timings, duration, demo),
        recommendation_card(level, fake, text_score),
        history_rows(history),
        analytics_panel(history),
        threat_center_panel(history),
        history
    )


def message_result(history, message_html):
    """Show a message in the banner area and clear the other cards."""
    return (
        message_html, "", "", "", "", "", "",
        history_rows(history),
        analytics_panel(history),
        threat_center_panel(history),
        history
    )


def error_banner(title, text):
    return f"""
    <div class="error-panel">
        <h3>{safe_text(title)}</h3>
        <p>{safe_text(text)}</p>
        <small>Check the terminal window for more information.</small>
    </div>
    """


# =========================================================
# LIVE AUDIO ANALYSIS
# =========================================================

def run_scan(audio_path, language, history):
    history = list(history or [])

    if not audio_path:
        return message_result(history, """
        <div class="error-panel">
            Please upload or record an audio file first.
        </div>
        """)

    try:
        started = time.perf_counter()

        # Layer 1: AI voice detection
        fake_score = float(voice_check(audio_path))
        fake_score = max(0.0, min(1.0, fake_score))
        after_voice = time.perf_counter()

        # Layer 2a: speech to text
        transcript = asr(audio_path, language)
        after_asr = time.perf_counter()

        # Layer 2b: scam text analysis
        text_score, hits = text_check(transcript)
        text_score = max(0.0, min(100.0, float(text_score)))
        after_text = time.perf_counter()

        # Fusion: one risk score and one verdict
        risk = float(combined_risk(fake_score, text_score))
        level = str(final_level(fake_score, text_score)).upper()
        finished = time.perf_counter()

        timings = {
            "voice_ms": (after_voice - started) * 1000,
            "asr_ms": (after_asr - after_voice) * 1000,
            "text_ms": (after_text - after_asr) * 1000,
            "total_ms": (finished - started) * 1000,
        }

        print(
            f"AWAAZ scan: voice={fake_score:.3f} text={text_score:.0f} "
            f"risk={risk:.0f} level={level} "
            f"total={timings['total_ms']:.0f}ms"
        )

        record = create_record(
            filename=Path(audio_path).name,
            language=language,
            fake_score=fake_score,
            text_score=text_score,
            risk_score=risk,
            verdict=level,
            transcript=transcript,
            hits=hits
        )

        history.append(record)
        history = history[-100:]

        plot_html, duration = make_audio_visual(audio_path)

        return render_result(
            level, risk, fake_score, text_score, transcript, hits,
            timings, duration, plot_html, language, history
        )

    except Exception as error:
        print("AWAAZ SCAN ERROR:", repr(error))

        return message_result(
            history, error_banner("Analysis Error", error)
        )


# =========================================================
# DEMO MODE
# =========================================================

def fuse_scores(fake_score, text_score):
    """Use the real scam_engine functions. Fall back to the project
    weights (35% voice, 65% text) so Demo Mode can never crash."""
    try:
        risk = float(combined_risk(fake_score, text_score))
        level = str(final_level(fake_score, text_score)).upper()
    except Exception as error:
        print("AWAAZ DEMO fallback scoring:", repr(error))
        risk = 0.35 * fake_score * 100 + 0.65 * text_score
        level = "RED" if risk >= 65 else "YELLOW" if risk >= 30 else "GREEN"

    return risk, level


def run_demo(key, history):
    """One-click demo. Returns the 11 result values plus the audio file."""
    history = list(history or [])

    try:
        scenario = SCENARIOS[key]

        fake_score = float(scenario["fake_score"])
        text_score = float(scenario["text_score"])
        hits = scenario["hits"]
        transcript = scenario["transcript"]
        language = scenario["language"]

        risk, level = fuse_scores(fake_score, text_score)

        record = create_record(
            filename="[DEMO] " + scenario["title"],
            language=language,
            fake_score=fake_score,
            text_score=text_score,
            risk_score=risk,
            verdict=level,
            transcript=transcript,
            hits=hits
        )

        history.append(record)
        history = history[-100:]

        audio_path = find_demo_audio(key)

        if audio_path:
            plot_html, duration = make_audio_visual(audio_path)
        else:
            plot_html, duration = plot_message(
                "No sample audio found for this scenario. Put a file in "
                "the demo_audio or sample folder to see its waveform."
            ), None

        result = render_result(
            level, risk, fake_score, text_score, transcript, hits,
            None, duration, plot_html, language, history, demo=scenario
        )

        return result + (audio_path,)

    except Exception as error:
        print("AWAAZ DEMO ERROR:", repr(error))

        return message_result(
            history, error_banner("Demo Error", error)
        ) + (None,)


def make_demo_handler(key):
    def handler(history):
        return run_demo(key, history)

    return handler


def warm_up_models():
    """Run both models once on 1 second of silence.

    This loads the models before the first real scan, so the first click
    is not slow. It runs before the page opens, so it cannot clash with
    a user click.
    """
    path = Path(tempfile.gettempdir()) / "awaaz_warmup.wav"

    try:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 16000)

        voice_check(str(path))
        asr(str(path), "english")
        print("AWAAZ: models are warm and ready.")

    except Exception as error:
        print("AWAAZ warm-up skipped:", repr(error))

    finally:
        try:
            path.unlink()
        except OSError:
            pass


# =========================================================
# GRADIO APPLICATION
# =========================================================

DEMO_PANEL = """
<div class="aw-demo-panel">
    <div class="aw-eyebrow">DEMO MODE · ONE-CLICK SCENARIOS</div>
    <p>
        Click a scenario to fill the whole dashboard instantly. Demo scores
        are preset, so they work even if the microphone, the internet or the
        models are slow. Press ANALYZE AUDIO afterwards to run the live
        pipeline on the loaded sample file.
    </p>
</div>
"""

with gr.Blocks(
    title="Awaaz | AI Voice Scam Shield"
) as demo:

    scan_history = gr.State([])

    demo.load(fn=None, js=BACKGROUND_JS)

    # TOP NAVBAR
    gr.HTML("""
    <header class="app-navbar">
        <div class="brand-mark">
            AWAAZ <span>AI</span>
        </div>
        <div class="navbar-caption">
            VOICE SCAM INTELLIGENCE SYSTEM
        </div>
        <div class="navbar-indicator">
            <span class="status-dot"></span> SYSTEM READY
        </div>
    </header>
    """)

    with gr.Tabs():

        # HOME
        with gr.Tab("Home"):
            gr.HTML(HOME)

        # SCAN
        with gr.Tab("Scan"):
            gr.HTML("""
            <div class="screen-heading">
                <div class="section-eyebrow">AUDIO SECURITY</div>
                <h2>Audio Security Scanner</h2>
                <p>Upload or record audio to check the voice and the spoken content.</p>
            </div>
            """)

            # Demo Mode
            gr.HTML(DEMO_PANEL)

            demo_buttons = {}

            with gr.Row():
                for demo_key, scenario in SCENARIOS.items():
                    demo_buttons[demo_key] = gr.Button(
                        scenario["button"],
                        variant="secondary"
                    )

            # Final verdict banner
            verdict_output = gr.HTML(value=idle_banner())

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    audio_input = gr.Audio(
                        label="Upload or Record Audio",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )

                    language_input = gr.Dropdown(
                        choices=["urdu", "english"],
                        value="urdu",
                        label="Speech Language"
                    )

                    scan_button = gr.Button(
                        "ANALYZE AUDIO",
                        variant="primary"
                    )

                with gr.Column(scale=1, min_width=280):
                    score_output = gr.HTML()

            # Layer 1 and Layer 2 breakdown
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    layer1_output = gr.HTML()

                with gr.Column(scale=1, min_width=280):
                    layer2_output = gr.HTML()

            plot_output = gr.HTML()
            performance_output = gr.HTML()
            guidance_output = gr.HTML()

        # THREAT CENTER
        with gr.Tab("Threat Center"):
            gr.HTML("""
            <div class="screen-heading">
                <div class="section-eyebrow">SECURITY OPERATIONS</div>
                <h2>Threat Intelligence Center</h2>
                <p>Review alerts and risk trends from this app session.</p>
            </div>
            """)

            threat_output = gr.HTML(
                value=threat_center_panel([])
            )

        # ANALYTICS
        with gr.Tab("Analytics"):
            gr.HTML("""
            <div class="screen-heading">
                <div class="section-eyebrow">DATA INSIGHTS</div>
                <h2>Scan Analytics</h2>
                <p>Summary of audio scans completed in this session.</p>
            </div>
            """)

            analytics_output = gr.HTML(
                value=analytics_panel([])
            )

            refresh_button = gr.Button(
                "↻ REFRESH ANALYTICS",
                variant="secondary"
            )

        # HOW IT WORKS
        with gr.Tab("How it works"):
            gr.HTML(HOW)

        # TEST RESULTS
        with gr.Tab("Test results"):
            gr.HTML("""
            <div class="screen-heading">
                <div class="section-eyebrow">MODEL EVALUATION</div>
                <h2>Previous Model Testing</h2>
            </div>

            <div class="result-card test-result-card">
                <div class="section-eyebrow">PRELIMINARY DATASET EVALUATION</div>
                <h3>Voice Authenticity Detection</h3>
                <div class="test-score">73.02%</div>
                <p>Accuracy: 92 correct classifications out of 126 audio files.</p>

                <div class="metric-track">
                    <div class="metric-fill voice-fill" style="width:73.02%"></div>
                </div>

                <div class="pattern-row">
                    <span>Precision</span><b>97.67%</b>
                </div>
                <div class="pattern-row">
                    <span>Recall</span><b>56.00%</b>
                </div>
                <div class="pattern-row">
                    <span>F1 Score</span><b>71.19%</b>
                </div>
                <div class="pattern-row">
                    <span>Specificity</span><b>98.04%</b>
                </div>

                <div class="divider"></div>

                <h3>Scam Content Detection</h3>
                <div class="test-score">75.40%</div>
                <div class="metric-track">
                    <div class="metric-fill text-fill" style="width:75.40%"></div>
                </div>

                <div class="pattern-row">
                    <span>Precision</span><b>87.10%</b>
                </div>
                <div class="pattern-row">
                    <span>Recall</span><b>50.00%</b>
                </div>
                <div class="pattern-row">
                    <span>F1 Score</span><b>63.53%</b>
                </div>
                <div class="pattern-row">
                    <span>Specificity</span><b>94.44%</b>
                </div>

                <div class="divider"></div>

                <h3>Combined Risk Assessment</h3>
                <div class="test-score">63.49%</div>
                <div class="metric-track">
                    <div class="metric-fill voice-fill" style="width:63.49%"></div>
                </div>

                <p class="muted-text">
                    Exact GREEN/YELLOW/RED classification: 70 out of 126
                    samples (55.56%).
                </p>

                <div class="notice-panel">
                    <b>Evaluation limitation:</b>
                    These are preliminary results from the current project
                    dataset. The dataset is small and contains possible
                    source-format and duration differences between classes.
                    These results are not independent real-world accuracy
                    measurements and should not be presented as such.
                </div>
            </div>
            """)

            gr.Markdown("### Live Session Scan History")

            history_table = gr.Dataframe(
                headers=[
                    "Time",
                    "Audio File",
                    "Verdict",
                    "Risk Score"
                ],
                datatype=["str", "str", "str", "number"],
                value=[],
                interactive=False,
                label="Recent Scans"
            )

            gr.Markdown(
                "Scan records are stored in the current app session. "
                "They are not permanently saved after the session ends."
            )

    # =====================================================
    # BUTTON CONNECTIONS
    # =====================================================

    result_outputs = [
        verdict_output,
        score_output,
        layer1_output,
        layer2_output,
        plot_output,
        performance_output,
        guidance_output,
        history_table,
        analytics_output,
        threat_output,
        scan_history
    ]

    scan_button.click(
        fn=run_scan,
        inputs=[
            audio_input,
            language_input,
            scan_history
        ],
        outputs=result_outputs,
        show_progress="full"
    )

    # Demo buttons also load the sample audio into the audio box.
    for demo_key, demo_button in demo_buttons.items():
        demo_button.click(
            fn=make_demo_handler(demo_key),
            inputs=[scan_history],
            outputs=result_outputs + [audio_input],
            show_progress="minimal"
        )

    refresh_button.click(
        fn=refresh_analytics,
        inputs=[scan_history],
        outputs=[
            analytics_output,
            history_table,
            threat_output
        ]
    )


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":
    # Load the models once before the page opens (set AWAAZ_WARMUP=0 to skip).
    if os.environ.get("AWAAZ_WARMUP", "1") != "0":
        print("AWAAZ: loading models, please wait...")
        warm_up_models()

    in_colab = "google.colab" in sys.modules

    demo.queue()

    demo.launch(
        server_name=os.environ.get("AWAAZ_HOST", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"),
        server_port=int(os.environ.get("AWAAZ_PORT", "7860")),
        show_error=True,
        css=CSS,
        theme=THEME,
        inbrowser=not in_colab,
        share=in_colab or os.environ.get("AWAAZ_SHARE") == "1",
        allowed_paths=[str(BASE_DIR)]
    )