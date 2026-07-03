"""Fusion / escalation layer.

SAFETY INVARIANT (architecture §7, refinement #2): no single stage
auto-dispatches. This layer combines cue word + breathing + (externally) the
passenger's response to a prompt, and only escalates on corroboration.
"""
from __future__ import annotations

from tash.audio import config
from tash.audio.contracts import (
    BreathingEstimate,
    BreathingState,
    CueWordEvent,
    EscalationLevel,
    FusionDecision,
)


class FusionEngine:
    def decide(
        self,
        ts: float,
        cue_word: CueWordEvent | None,
        breathing: BreathingEstimate | None,
        passenger_responded: bool | None = None,
    ) -> FusionDecision:
        reasons: list[str] = []
        level = EscalationLevel.NONE

        agonal = (
            breathing is not None
            and breathing.state
            in (BreathingState.AGONAL_SUSPECT, BreathingState.APNEA)
        )

        if cue_word is not None:
            reasons.append(f"cue word '{cue_word.keyword}' detected")
            level = EscalationLevel.CONFIRM      # never auto-dispatch on a word

        if agonal:
            reasons.append(f"breathing state={breathing.state.value} (advisory)")
            level = max(level, EscalationLevel.CONFIRM, key=_rank)

        # Corroboration: distress cue AND respiratory distress AND no response.
        if cue_word is not None and agonal:
            reasons.append("corroborated: cue word + respiratory distress")
            level = EscalationLevel.ESCALATE
        elif level is EscalationLevel.CONFIRM and passenger_responded is False:
            reasons.append("no passenger response to confirmation prompt")
            level = EscalationLevel.ESCALATE

        if level is EscalationLevel.NONE and breathing is not None:
            level = EscalationLevel.MONITOR

        return FusionDecision(
            ts=ts,
            level=level,
            reasons=reasons,
            cue_word=cue_word,
            breathing=breathing,
        )


_ORDER = {
    EscalationLevel.NONE: 0,
    EscalationLevel.MONITOR: 1,
    EscalationLevel.CONFIRM: 2,
    EscalationLevel.ESCALATE: 3,
}


def _rank(level: EscalationLevel) -> int:
    return _ORDER[level]
