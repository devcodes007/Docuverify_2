"""
Post-generation retry policy, kept as a small pure function so it's
independently testable and the orchestrator doesn't need to embed this
state machine inline.

SUPPORTED                -> return the answer.
CONTRADICTED/UNSUPPORTED -> retry once (re-retrieve + regenerate + re-verify).
Still not SUPPORTED after the retry -> explicit refusal, never hallucinate.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.models.schemas import GroundednessLabel


class RetryDecision(str, Enum):
    ACCEPT = "ACCEPT"
    RETRY = "RETRY"
    REFUSE = "REFUSE"


REFUSAL_MESSAGE = (
    "I don't have sufficient evidence in the indexed documentation to answer "
    "this reliably. Please rephrase the question or check the official docs directly."
)


@dataclass
class RetryState:
    verification_attempts: int = 0


def decide(label: GroundednessLabel, state: RetryState, max_verification_attempts: int = 2) -> RetryDecision:
    if label == GroundednessLabel.SUPPORTED:
        return RetryDecision.ACCEPT
    if state.verification_attempts + 1 >= max_verification_attempts:
        return RetryDecision.REFUSE
    return RetryDecision.RETRY
