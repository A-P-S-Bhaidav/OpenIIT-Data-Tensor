"""FastAPI service exposing the PTP credibility engine.

Endpoints
---------
POST /v1/calls                         open a live call session
POST /v1/calls/{session_id}/turns      push one transcript turn, get live score + prompts
POST /v1/calls/{session_id}/terms      promise terms captured on the desktop / by the bot
POST /v1/calls/{session_id}/end        final score, disposition, schedule, reason codes
POST /v1/calls/{session_id}/events     post-call engagement event -> updated score
GET  /v1/health                        liveness and model versions

Run:  uvicorn ptp_credibility.service.api:app --port 8080   (with src/ on PYTHONPATH)
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import CredibilityEngine

app = FastAPI(title="CN PTP Credibility Service", version="1.0.0",
              description="Real-time promise-to-pay credibility scoring, typing, prompting and disposition.")
_engine: Optional[CredibilityEngine] = None


def engine() -> CredibilityEngine:
    global _engine
    if _engine is None:
        _engine = CredibilityEngine()
    return _engine


class StartCall(BaseModel):
    ptp_id: str = Field(..., description="PTP / call context key in the feature store")
    voice_consent: bool = Field(True, description="DPDP consent for voice-derived (prosodic) features")
    captured_ts: Optional[str] = None


class TurnIn(BaseModel):
    speaker: str = Field(..., pattern="^(agent|bot|customer)$")
    text: str
    pause_before_ms: float = 0.0
    start_ms: float = 0.0
    end_ms: float = 0.0
    asr_confidence: float = 1.0


class TermsIn(BaseModel):
    promised_date: str
    promised_amount: float


class EventIn(BaseModel):
    event_type: str = Field(..., pattern="^(payment_link_opened|confirmation_reply_yes|confirmation_reply_no|"
                                         "inbound_call_from_customer|whatsapp_confirmation_sent)$")
    hours_after_call: float = 0.0


def _session(sid: str):
    s = engine().sessions.get(sid)
    if s is None:
        raise HTTPException(404, f"Unknown session {sid}")
    return s


@app.get("/v1/health")
def health():
    e = engine()
    return {"status": "ok", "versions": e.versions, "thresholds": e.thresholds}


@app.post("/v1/calls")
def start_call(body: StartCall):
    try:
        s = engine().start(body.ptp_id, body.voice_consent, body.captured_ts)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return {"session_id": s.session_id, "voice_features_used": s.voice_consent}


@app.post("/v1/calls/{session_id}/turns")
def add_turn(session_id: str, body: TurnIn):
    s = _session(session_id)
    if s.closed:
        raise HTTPException(409, "Session already closed")
    return s.add_turn(body.speaker, body.text, body.pause_before_ms, body.start_ms, body.end_ms, body.asr_confidence)


@app.post("/v1/calls/{session_id}/terms")
def set_terms(session_id: str, body: TermsIn):
    s = _session(session_id)
    s.set_terms(body.promised_date, body.promised_amount)
    return {"ok": True}


@app.post("/v1/calls/{session_id}/end")
def end_call(session_id: str):
    return _session(session_id).end_call()


@app.post("/v1/calls/{session_id}/events")
def post_event(session_id: str, body: EventIn):
    try:
        return _session(session_id).post_call_event(body.event_type, body.hours_after_call)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
